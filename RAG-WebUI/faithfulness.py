# -*- coding: utf-8 -*-
"""断言级忠实度评测：把答案拆成原子断言，逐条判定是否被引用的 chunk 支持。

这是 README「下一步」里列的那一项。为什么需要它：
现有的生成层指标（json_valid_rate / answerable_accuracy / 幻觉率）只测
「答了没有」和「格式对不对」，**测不出答错了**。实测漏掉过把
「code-based Grader 成本极低、毫秒级」答成「成本较高、速度较慢」这种完全颠倒的答案。

方法（两步，都是 LLM 调用）：
  1) 分解：把答案拆成原子断言，每条带它引用的来源编号。
  2) 判定：**每次只判一条断言**（一次一个维度，避免裁判混淆），
     给出 SUPPORTED / CONTRADICTED / UNSUPPORTED + 证据原文引用。
     设了「资料不足就判 UNSUPPORTED」的逃生门，不允许裁判勉强打分。

用法：
  python faithfulness.py --check          # 只在已知对错的 10 条上验证判据（先跑这个）
  python faithfulness.py                  # 全量跑两个库
  python faithfulness.py --only ai_prep
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "RAG知识库Agent" / "demo"
sys.path.insert(0, str(DEMO))
import kb_agent as K  # noqa: E402

TOPK = 3
RERANK = "max"
POOL = 40
ALPHA = 0.7

# ---------------------------------------------------------------------------
# 分解提示词
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = """你是答案审核助手。把给定的【答案】拆成若干条**原子断言**。

规则：
1. 一条断言只包含一个可独立验证的事实。含「和」「并且」「同时」的复合句要拆开。
2. 每条断言标注它在【答案】中引用的来源编号（形如 [1] [2]）。
   如果那条断言没有标注任何编号，citations 留空数组。
3. 只拆事实性断言。纯过渡语、连接词、重复表述不要单独成条。
4. 最多拆 8 条，优先保留最关键的。
5. 只输出 JSON，不要输出解释或 Markdown 代码块。

输出 JSON 格式：
{"assertions": [{"text": "断言原文（可精简，但不得改变含义）", "citations": [1]}]}
"""

# ---------------------------------------------------------------------------
# 判定提示词（一次只判一条，符合「One Dimension Per Call」最佳实践）
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """你是严格的事实核查员。判断【断言】是否被【资料】支持。

判定标准（必须严格执行）：
- SUPPORTED：资料中明确陈述了该断言，或其直接逻辑蕴含。你需要能引出一句原文作为证据。
- CONTRADICTED：资料明确说了与断言相反的内容。
- UNSUPPORTED：资料里没有提到，或不足以判断。

硬性规则：
1. 只依据【资料】判断，不得使用你自己的知识补充。
2. **如果资料不足以判断，必须选 UNSUPPORTED，绝对不要勉强给 SUPPORTED。**
3. SUPPORTED 时必须在 evidence 里原样引用资料中的一句话作为证据。
4. 只输出 JSON，不要输出解释或 Markdown 代码块。

输出 JSON 格式：
{"verdict": "SUPPORTED|CONTRADICTED|UNSUPPORTED", "evidence": "资料原文引用，或空字符串", "reason": "一句话"}
"""


def _parse_obj(raw: str) -> dict:
    """从模型输出里取 JSON 对象。

    注意：不能用 kb_agent.parse_json_lenient()——那是 RAG 答案专用的校验器，
    强制要求 answerable/answer/citations/missing 四个字段，
    拿来校验分解与判定结果会直接报「缺少字段 answerable」。
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|```$", "", s).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # 模型偶尔会在 JSON 前后加解释，抓第一个花括号块兜底
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("输出不是 JSON 对象")
    return obj


def _chat_json(messages: list[dict], retries: int = 2) -> dict:
    """调 chat 并解析 JSON，失败重试。"""
    last = None
    for _ in range(retries + 1):
        try:
            return _parse_obj(K.chat(messages, temperature=0.0))
        except Exception as e:                              # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"取 JSON 失败: {last}")


def decompose(answer: str) -> list[dict]:
    """把答案拆成原子断言。"""
    obj = _chat_json([
        {"role": "system", "content": DECOMPOSE_PROMPT},
        {"role": "user", "content": f"【答案】\n{answer}"},
    ])
    out = []
    for a in (obj.get("assertions") or [])[:8]:
        if isinstance(a, dict) and a.get("text"):
            cits = a.get("citations")
            out.append({"text": str(a["text"]),
                        "citations": [c for c in cits if isinstance(c, int)]
                        if isinstance(cits, list) else []})
    return out


def build_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] 来源：{h['doc']} / {h['cid']}\n{h['text']}")
    return "<资料>\n" + "\n\n".join(lines) + "\n</资料>"


def judge(assertion: dict, ctx: str) -> dict:
    """判定一条断言。返回 {verdict, evidence, reason}。"""
    obj = _chat_json([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"{ctx}\n\n【断言】\n{assertion['text']}"},
    ])
    v = str(obj.get("verdict", "")).upper().strip()
    if v not in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED"):
        v = "UNSUPPORTED"
    return {"verdict": v,
            "evidence": str(obj.get("evidence") or "")[:200],
            "reason": str(obj.get("reason") or "")[:200]}


def check_answer(q: str, answer: str, hits: list[dict],
                 store: dict | None = None) -> dict:
    """对一条答案做断言级忠实度检查。

    store 用来缓存分解与判定结果，避免重复调用（同一答案只判一次）。
    """
    store = store if store is not None else {}
    key = f"{q}\x00{answer}"
    if key in store:
        return store[key]

    assertions = decompose(answer)
    ctx = build_context(hits)
    verdicts = []
    for a in assertions:
        r = judge(a, ctx)
        # 引用编号是否真的存在（防模型编造 [5] 而只提供了 3 条资料）
        bad_cit = [c for c in a["citations"] if not (1 <= c <= len(hits))]
        r.update({"text": a["text"], "citations": a["citations"],
                  "fake_citations": bad_cit})
        verdicts.append(r)

    n = len(verdicts)
    n_sup = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
    n_con = sum(1 for v in verdicts if v["verdict"] == "CONTRADICTED")
    n_uns = sum(1 for v in verdicts if v["verdict"] == "UNSUPPORTED")
    res = {
        "q": q, "n_assertions": n,
        "n_supported": n_sup, "n_contradicted": n_con, "n_unsupported": n_uns,
        "faithfulness": round(n_sup / n, 3) if n else None,
        "has_contradiction": n_con > 0,
        "assertions": verdicts,
    }
    store[key] = res
    return res


# ---------------------------------------------------------------------------
# 第一批验证：10 条已知对错的样本
# 前 5 条是检索漏召的（含 2 条我人工确认答错的），后 5 条是命中的正常答案
# ---------------------------------------------------------------------------
KNOWN = [
    # (库, 问题, 人工判定, 说明)
    ("agent_eval", "评测 Agent 时，Grader 在成本、速度和确定性上怎么权衡？",
     "答错", "模型说 code-based 成本高速度慢，文档写的是极低成本毫秒级 —— 完全颠倒"),
    ("ai_prep", "用户问法和文档里的说法完全不一样，怎么提高命中率？",
     "答错", "答的是表格解析/权限过滤，与查询改写/HyDE 无关"),
    ("ai_prep", "模型离线指标很好看，一上线就掉，通常是什么原因？",
     "答偏", "泛泛而谈，没提数据泄漏与分布漂移"),
    ("agent_eval", "最小的可用 Benchmark 要多少个 task？怎么分配？",
     "答对", "内容在 #12，答案正确"),
    ("ai_prep", "手撕自注意力时，哪些细节最容易写错？",
     "答对", "矩阵维度/缩放因子/softmax/KQV，基本正确"),
    ("agent_eval", "Transcript 和 Outcome 这两个概念的区别是什么？", "答对", "命中 gold"),
    ("agent_eval", "pass@k 和 pass^k 分别表示什么？", "答对", "命中 gold"),
    ("agent_eval", "三层质量门分别测什么、什么频率？", "答对", "命中 gold"),
    ("ai_prep", "KV Cache 的显存怎么估算？写出公式。", "答对", "命中 gold，公式正确"),
    ("ai_prep", "混合检索里 RRF 的公式是什么？k 取多少？", "答对", "命中 gold，公式正确"),
]


def run_check() -> None:
    """在已知对错的样本上验证判据是否可信。"""
    print("=" * 78)
    print("判据验证：10 条已知对错的样本")
    print("（若把「答对」判成低分，说明判据太严；若把「答错」判成高分，说明判据太松）\n")
    store: dict = {}
    report = json.loads((HERE / "kb" / "_report_index.json").read_text("utf-8")) \
        if (HERE / "kb" / "_report_index.json").exists() else None

    # 收集两个库的生成结果
    rows_by_kb = {}
    for kb_id in ("ai_prep", "agent_eval"):
        d = json.loads((HERE / "kb" / kb_id / "eval" / "report_generation.json")
                       .read_text("utf-8"))
        idx = K.Retriever(HERE / "kb" / kb_id / "index")
        chunks = {c["cid"]: c for c in idx.chunks}
        rows_by_kb[kb_id] = (d["cases"], chunks)

    results = []
    for kb_id, q, human, note in KNOWN:
        rows, chunks = rows_by_kb[kb_id]
        row = next((x for x in rows if x["q"] == q), None)
        if row is None:
            print(f"  !! 找不到题：{q}")
            continue
        hits = []
        for cid in row["hits"]:
            c = chunks.get(cid)
            if c:
                hits.append({"cid": c["cid"], "doc": c["doc"], "text": c["text"]})
        t0 = time.time()
        res = check_answer(q, row["answer"] or "", hits, store)
        results.append((human, note, res))
        print(f"  [{human}] 忠实度={res['faithfulness']}  "
              f"支持 {res['n_supported']} / 矛盾 {res['n_contradicted']} / "
              f"无据 {res['n_unsupported']}  ({time.time()-t0:.1f}s)")
        print(f"       Q: {q[:52]}")
        for v in res["assertions"]:
            mark = {"SUPPORTED": "✓", "CONTRADICTED": "✗矛盾",
                    "UNSUPPORTED": "?无据"}[v["verdict"]]
            print(f"        {mark} {v['text'][:66]}")
            if v["verdict"] == "CONTRADICTED":
                print(f"             理由: {v['reason'][:80]}")
        print()

    # 汇总：判据是否分得开「答对」与「答错」
    good = [r for h, _, r in results if h == "答对"]
    bad = [r for h, _, r in results if h in ("答错", "答偏")]
    import statistics
    if good and bad:
        gm = statistics.mean(r["faithfulness"] for r in good)
        bm = statistics.mean(r["faithfulness"] for r in bad)
        print("=" * 78)
        print(f"判据区分度：答对组平均忠实度 {gm:.3f}（{len(good)} 条） | "
              f"答错/答偏组 {bm:.3f}（{len(bad)} 条） | 差距 {gm-bm:+.3f}")
        contra = [r for r in bad if r["has_contradiction"]]
        print(f"答错组中检出「断言与资料矛盾」的：{len(contra)}/{len(bad)} 条")
        print(f"答对组中被误报「矛盾」的："
              f"{sum(1 for r in good if r['has_contradiction'])}/{len(good)} 条"
              f"  ← 应为 0，否则判据太严")


def run_full(only: str | None) -> None:
    """全量跑：用已有的生成报告，不重新调生成模型。"""
    store: dict = {}
    for kb_id in ("ai_prep", "agent_eval"):
        if only and kb_id != only:
            continue
        rep_f = HERE / "kb" / kb_id / "eval" / "report_generation.json"
        d = json.loads(rep_f.read_text("utf-8"))
        rows = d["cases"]
        idx = K.Retriever(HERE / "kb" / kb_id / "index")
        chunks = {c["cid"]: c for c in idx.chunks}
        print("=" * 78)
        print(f"[{kb_id}] 断言级忠实度：{len(rows)} 条")

        out = []
        for i, row in enumerate(rows, 1):
            hits = [{"cid": chunks[c]["cid"], "doc": chunks[c]["doc"],
                     "text": chunks[c]["text"]}
                    for c in row["hits"] if c in chunks]
            res = check_answer(row["q"], row["answer"] or "", hits, store)
            out.append({"q": row["q"], "cat": row["cat"],
                        "retrieval_hit": row["retrieval_hit"],
                        "judged_answerable": row["judged_answerable"],
                        "answer": row["answer"],
                        "citations": row["citations"],
                        "hits": row["hits"],
                        **{k: res[k] for k in ("n_assertions", "n_supported",
                                               "n_contradicted", "n_unsupported",
                                               "faithfulness", "has_contradiction")},
                        "assertions": res["assertions"]})
            if i % 5 == 0 or i == len(rows):
                print(f"    ... {i}/{len(rows)}")

        # 只统计真作答的（拒答题没有断言要核）
        answered = [x for x in out if x["judged_answerable"] and x["n_assertions"]]
        import statistics
        fa = [x["faithfulness"] for x in answered]
        metrics = {
            "n_cases": len(rows),
            "n_answered_with_assertions": len(answered),
            "faithfulness_mean": round(statistics.mean(fa), 4) if fa else None,
            "faithfulness_median": round(statistics.median(fa), 4) if fa else None,
            "n_with_contradiction": sum(1 for x in answered if x["has_contradiction"]),
            "n_unsupported_any": sum(1 for x in answered if x["n_unsupported"]),
            "n_fake_citations": sum(1 for x in out for a in x["assertions"]
                                    if a.get("fake_citations")),
            "total_assertions": sum(x["n_assertions"] for x in out),
        }
        (HERE / "kb" / kb_id / "eval" / "report_faithfulness.json").write_text(
            json.dumps({"kb": kb_id, "metrics": metrics, "cases": out},
                       ensure_ascii=False, indent=1), encoding="utf-8")

        print(f"\n  === 断言级忠实度指标 ===")
        for k, v in metrics.items():
            if k != "n_cases":
                print(f"    {k:36} {v}")
        bad = sorted(answered, key=lambda x: x["faithfulness"])[:5]
        print(f"\n  忠实度最低 5 条：")
        for x in bad:
            print(f"    {x['faithfulness']}  矛盾{x['n_contradicted']} 无据{x['n_unsupported']}"
                  f"  {x['q'][:44]}")
        con = [x for x in answered if x["has_contradiction"]]
        if con:
            print(f"\n  ⚠ 检出与资料矛盾的答案 {len(con)} 条：")
            for x in con:
                print(f"    Q: {x['q'][:50]}")
                for a in x["assertions"]:
                    if a["verdict"] == "CONTRADICTED":
                        print(f"       断言: {a['text'][:70]}")
                        print(f"       理由: {a['reason'][:90]}")
        print(f"\n  报告已写入 kb/{kb_id}/eval/report_faithfulness.json\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="断言级忠实度评测")
    ap.add_argument("--check", action="store_true", help="只在已知对错的 10 条上验证判据")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.check:
        run_check()
    else:
        run_full(args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
