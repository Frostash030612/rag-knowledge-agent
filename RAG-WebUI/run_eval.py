# -*- coding: utf-8 -*-
"""跑全部知识库的评测指标。

默认只跑检索层（免费）。加 --gen 才跑生成层（会调用 LLM，消耗额度）。

检索层复用 demo 的 kb_agent.evaluate(retrieval_only=True)，保证与 demo 同口径。
生成层用 kb_agent.ask_stream() 跑，这样能顺带量到 TTFT（首字延迟）——
那是流式改造后最该量化、而 demo 的 evaluate() 没有覆盖的指标。

用法：
  python run_eval.py                 # 只跑检索层
  python run_eval.py --gen           # 检索层 + 生成层
  python run_eval.py --gen --only ai_prep
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
# 答案覆盖度：一个「烟雾探测器」，不是忠实度指标
#
# 现有生成层指标（json_valid_rate / answerable_accuracy / 幻觉率）只测
# 「答了没有」，不测「答得对不对」——实测确实漏掉了答反的答案
# （如把 code-based Grader 说成「成本高速度慢」，而文档写的是「极低成本毫秒级」）。
#
# 这里用 eval 集的 ref（参考答案要点）做词面覆盖，能抓出「答案与 ref 完全不相干」，
# **但抓不住语义矛盾，也会低估正确但措辞不同的答案**
# （例如答「score(d) = Σ 1/(k+rank)」而 ref 里是分词碎片，覆盖率只有 0.12）。
# 所以它只适合当告警，不能当作指标写进结论。真正的忠实度评测需要
# 断言级引用校验或 LLM-as-judge，见 README 的「下一步」。
# ---------------------------------------------------------------------------
ENT = re.compile(r"[A-Za-z][A-Za-z0-9\-\./^]{2,}|[\u4e00-\u9fff]{2,}")
STOP = re.compile(r"^[的是什么怎么为吗呢了和与或及在把对被让使从到等这那很就都也还要会能可"
                  r"用有不上下来去个我们你他它其中一种常见包括主要因为所以如果则需应当]+$")


def ref_keys(s: str) -> set[str]:
    out: set[str] = set()
    for tok in ENT.findall(s):
        if STOP.match(tok):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\./^]*", tok):
            out.add(tok.lower())
        else:
            for n in (4, 3, 2):
                for i in range(len(tok) - n + 1):
                    g = tok[i:i + n]
                    if not STOP.match(g):
                        out.add(g)
    return out


def coverage(answer: str, ref: str) -> float | None:
    keys = ref_keys(ref)
    if not keys:
        return None
    a = (answer or "").lower()
    return round(sum(1 for k in keys if k in a) / len(keys), 3)


def eval_retrieval(kb: dict, idx: Path, ev: Path) -> dict:
    cases = K.load_eval_set(ev)
    n_gold = sum(1 for c in cases if c.get("gold_cids"))
    n_unans = len(cases) - n_gold
    n_chunks = json.loads((idx / "meta.json").read_text("utf-8"))["n_chunks"]

    print("=" * 74)
    print(f"{kb['name']}（{kb['id']}）：{len(cases)} 条 "
          f"= {n_gold} 可答 + {n_unans} 不可答 | 语料 {n_chunks} chunks")
    rep = K.evaluate(idx, ev, HERE / "kb" / kb["id"] / "eval" / "report_retrieval.json",
                     top_k=TOPK, modes=["vector", "bm25", "hybrid"],
                     retrieval_only=True, rerank=RERANK, rerank_pool=POOL, alpha=ALPHA)
    print(f"  {'模式':<10}{'Recall@3':>10}{'MRR':>10}")
    for mode, m in rep["modes"].items():
        print(f"  {mode:<10}{m['recall@3']:>10.4f}{m['mrr']:>10.4f}")

    # 分类别
    r = K.Retriever(idx)
    tot = {k: 0 for k in "ABC"}
    for c in cases:
        if c.get("gold_cids"):
            tot[c["cat"][0]] += 1
    print(f"\n  分类别 Recall@3（A 精确匹配 / B 语义泛化 / C 聚合）：")
    print(f"  {'模式':<10}" + "".join(f"{k + '类':>10}" for k in "ABC"))
    for mode in ("vector", "bm25", "hybrid"):
        hit = {k: 0 for k in "ABC"}
        for c in cases:
            gold = set(c.get("gold_cids") or [])
            if not gold:
                continue
            got = {r.chunks[i]["cid"] for i, _ in
                   r.search(c["q"], top_k=TOPK, mode=mode, rerank=RERANK,
                            rerank_pool=POOL, alpha=ALPHA)}
            if gold & got:
                hit[c["cat"][0]] += 1
        print(f"  {mode:<10}" + "".join(
            f"{(hit[k] / tot[k] if tot[k] else 0):>10.3f}" for k in "ABC"))
    print()
    return rep


def eval_generation(kb: dict, idx: Path, ev: Path) -> dict:
    """生成层指标。用 ask_stream 以便同时量 TTFT。"""
    cases = K.load_eval_set(ev)
    r = K.Retriever(idx)
    out_path = HERE / "kb" / kb["id"] / "eval" / "report_generation.json"

    print("=" * 74)
    print(f"{kb['name']}（{kb['id']}）：生成层 {len(cases)} 条")

    rows = []
    n_json_ok = n_repairs = 0
    n_ans = n_ans_ok = 0
    n_unans = n_ans_when_unans = 0
    ttfts: list[float] = []
    totals: list[float] = []

    for i, c in enumerate(cases, 1):
        t0 = time.time()
        first = None
        text, res = "", None
        for kind, payload in K.ask_stream(c["q"], r, top_k=TOPK, mode="hybrid",
                                          rerank=RERANK, rerank_pool=POOL, alpha=ALPHA):
            if kind == "delta":
                if first is None:
                    first = time.time() - t0
                text += payload
            elif kind == "result":
                res = payload
        dt = time.time() - t0

        hit_cids = res.get("_hits", [])
        gold = set(c.get("gold_cids") or [])
        retrieval_hit = bool(gold & set(hit_cids))
        failed = bool(res.get("_failed"))
        judged_answerable = bool(res.get("answerable"))

        n_json_ok += 0 if failed else 1
        n_repairs += res.get("_repairs", 0)
        if c.get("answerable", True):
            n_ans += 1
            n_ans_ok += 1 if judged_answerable else 0
        else:
            n_unans += 1
            n_ans_when_unans += 1 if judged_answerable else 0
        if first is not None:
            ttfts.append(first)
        totals.append(dt)

        rows.append({
            "q": c["q"], "cat": c["cat"], "kb": kb["id"],
            "gold": sorted(gold), "hits": hit_cids, "retrieval_hit": retrieval_hit,
            "expected_answerable": c.get("answerable", True),
            "judged_answerable": judged_answerable,
            "answer": res.get("answer"), "citations": res.get("citations"),
            "missing": res.get("missing"), "repairs": res.get("_repairs", 0),
            "failed": failed, "ttft_s": round(first, 2) if first else None,
            "total_s": round(dt, 2),
            "ref": c.get("ref", ""),
            "answer_coverage": coverage(res.get("answer") or "", c.get("ref", ""))
            if c.get("answerable", True) else None,
        })
        if i % 10 == 0 or i == len(cases):
            print(f"    ... {i}/{len(cases)}  (最近一条 ttft={first and round(first,2)}s, "
                  f"total={dt:.1f}s)")

    import numpy as np
    n = len(cases)
    covs = [x["answer_coverage"] for x in rows if x["answer_coverage"] is not None]
    metrics = {
        "n_cases": n,
        "json_valid_rate": round(n_json_ok / n, 4),
        "answerable_accuracy": round(n_ans_ok / n_ans, 4) if n_ans else None,
        "false_answer_rate_on_unanswerable":
            round(n_ans_when_unans / n_unans, 4) if n_unans else None,
        "avg_repairs": round(n_repairs / n, 3),
        "ttft_median_s": round(float(np.median(ttfts)), 2) if ttfts else None,
        "ttft_p95_s": round(float(np.percentile(ttfts, 95)), 2) if ttfts else None,
        "latency_median_s": round(float(np.median(totals)), 2),
        "latency_p95_s": round(float(np.percentile(totals, 95)), 2),
        "latency_max_s": round(float(np.max(totals)), 2),
        # 见文件头部说明：烟雾探测器，不是忠实度指标，不要写进结论
        "answer_coverage_median": round(float(np.median(covs)), 3) if covs else None,
        "answer_coverage_zero_n": sum(1 for c in covs if c == 0),
    }
    out_path.write_text(json.dumps({"kb": kb["id"], "metrics": metrics, "cases": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n  === 生成层指标 ===")
    for k, v in metrics.items():
        if k != "n_cases":
            print(f"    {k:38} {v}")

    # 区分「正确拒绝」与「真问题」，口径与 demo 的 print_report 一致
    ok_refuse, bad = [], []
    for x in rows:
        if not x["expected_answerable"] and not x["judged_answerable"] and not x["failed"]:
            ok_refuse.append(x)
        elif x["failed"] or (x["gold"] and not x["retrieval_hit"]):
            bad.append(("检索漏召/输出非法", x))
        elif not x["expected_answerable"] and x["judged_answerable"]:
            bad.append(("幻觉：该拒答却作答", x))
    print(f"\n  正确拒绝（资料不足，符合预期）：{len(ok_refuse)} 条")
    for x in ok_refuse:
        print(f"    ✓ {x['q'][:46]}")
    print(f"\n  需要关注：{len(bad)} 条")
    for why, x in bad:
        print(f"    - [{why}] {x['q'][:44]}")

    # 低覆盖告警：可能是「答了但没答到点上」，也可能只是措辞不同，需人工看一眼
    low = [x for x in rows
           if x["answer_coverage"] is not None and x["answer_coverage"] < 0.15]
    if low:
        print(f"\n  ⚠ 答案与 ref 词面覆盖 <0.15（{len(low)} 条）——"
              f"只说明可能没答到点上，必须人工复核，不是错误判定：")
        for x in low:
            print(f"    - [{x['answer_coverage']}] {x['q'][:40]}")
            print(f"        ref: {(x.get('ref') or '')[:88]}")
            print(f"        答 : {(x['answer'] or '')[:88]}")
    print(f"\n  报告已写入 kb/{kb['id']}/eval/report_generation.json\n")
    return metrics


def eval_dir_of(kb: dict, here: Path = HERE) -> Path:
    """知识库的评测集目录。

    默认是 kb/<id>/eval（由 build_eval.py 生成）；如果注册表里显式给了
    eval_dir，就用它——dram 库没有自己的评测集，直接复用 demo 的 38 条
    （题面与文档一致、gold cid 完全相同，见变更记录 2026-10-14（六））。
    """
    if kb.get("eval_dir"):
        return (here / kb["eval_dir"]).resolve()
    return here / kb["docs"].replace("docs", "eval")


def main() -> int:
    ap = argparse.ArgumentParser(description="跑知识库评测指标")
    ap.add_argument("--gen", action="store_true", help="同时跑生成层（消耗额度）")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    reg = json.loads((HERE / "kb" / "registry.json").read_text("utf-8"))
    summary = {}
    for kb in reg["kbs"]:
        if args.only and kb["id"] != args.only:
            continue
        idx = HERE / kb["index"]
        ev = eval_dir_of(kb) / "eval_set.jsonl"
        if not ev.exists():
            print(f"[跳过] {kb['id']}：没有评测集（{ev}）\n")
            continue
        key = {"id": kb["id"], "name": kb["name"]}
        eval_retrieval(key, idx, ev)
        if args.gen:
            summary[kb["id"]] = eval_generation(key, idx, ev)

    if summary:
        print("=" * 74)
        print("生成层汇总")
        keys = ["json_valid_rate", "answerable_accuracy",
                "false_answer_rate_on_unanswerable", "avg_repairs",
                "ttft_median_s", "ttft_p95_s", "latency_median_s", "latency_p95_s"]
        print(f"  {'库':<12}" + "".join(f"{k[:14]:>16}" for k in keys))
        for kb_id, m in summary.items():
            print(f"  {kb_id:<12}" + "".join(f"{str(m.get(k)):>16}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
