# -*- coding: utf-8 -*-
"""答案正确性评测（LLM-as-judge）：检查回答是否真的答到了点上。

为什么需要它
------------
现有的生成层指标只测「答了没有」和「格式对不对」：
  json_valid_rate / answerable_accuracy / false_answer_rate_on_unanswerable
**它们测不出答错了。** 实测漏掉过把「code-based Grader 成本极低、毫秒级」
答成「成本较高、速度较慢」这种无依据的答案，那条的 answerable_accuracy 照样是 1.0。

我先后试了两种判据，在同一批 10 条已知对错的样本上对比（见 _compare_judges.py）：

| 判据 | 答对组 | 答错组 | 区分度 | 冤枉答对 | 漏放答错 |
| --- | --- | --- | --- | --- | --- |
| 断言级**忠实度** | 0.857 | 0.708 | +0.149 | 1/7 | 1/3 |
| **答案正确性**（本文件） | 0.929 | 0.667 | **+0.262** | 1/7 | 1/3 |

忠实度被淘汰的原因是它测错了对象：它判断「答案是否忠于给定上下文」，
而**检索到错误上下文时，基于它作答反而更「忠实」**——所以看不见答错。
它还因为要求字面证据，把措辞不同但正确的答案判成 0 分。
正确性判据拿答案与 eval 集的 ref 比语义，区分度是前者的 1.8 倍，且理由可读。

判据的已知边界
--------------
- **PARTIAL 的含义是「答对了但漏了要点」**，不是「答错」。所以汇总同时给出
  `wrong_rate = (PARTIAL + INCORRECT) / 总数`，看趋势以这个为准。
- 「答非所问但内容落在同一主题」的答案会被判 PARTIAL 而非 INCORRECT
  （实测那条「问法不同怎么提高命中率 → 答 RAG 工程实践」就是）。
- 少量正确答案可能因漏掉次要细节被判 PARTIAL，因此**不要用 1.0 作为及格线**。

用法：
  python accuracy_eval.py              # 全量跑两个库（用已有生成报告，不重新生成）
  python accuracy_eval.py --only ai_prep
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
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

ACCURACY_PROMPT_V1 = """你是严格的技术答案评分员。判断【考生答案】是否正确回答了【问题】。

你会拿到两份参考材料：
- 【参考答案要点】：出题人写的正确要点（权威依据）
- 【检索到的资料】：系统当时提供给考生的资料（可能不完整，甚至与问题无关）

评分等级（必须选一个）：
- CORRECT：答案与【参考答案要点】在实质上一致。措辞不同、更简略、少讲一个次要细节，都算 CORRECT。
- PARTIAL：答对了一部分，但遗漏关键要点或含次要错误。
- INCORRECT：答错了关键事实，答非所问，或答案内容在【参考答案要点】中找不到依据。
- REFUSED：答案表示资料不足／拒绝作答。

硬性规则：
1. 以【参考答案要点】为准，不要用你自己的知识去补充或推翻它。
2. 如果答案讲的内容在【参考答案要点】里完全没有、也不等价，必须判 INCORRECT，哪怕它看起来很专业。
3. 不要因为措辞与参考要点不同就判错——只要意思一致就是 CORRECT。
4. 关键词命中检查：先列出【参考答案要点】里的关键概念，再逐一到【考生答案】里找。
   只要有一个关键概念在答案里完全找不到对应表述，等级不得高于 PARTIAL；
   若有多个关键概念缺失，或答案讲的是另一套内容，判 INCORRECT。
5. 只输出 JSON，不要解释、不要 Markdown 代码块。

输出 JSON：
{"grade": "CORRECT|PARTIAL|INCORRECT|REFUSED", "reason": "一句话",
 "missing": "漏掉的关键概念，逗号分隔；没有则空字符串"}
"""

ACCURACY_PROMPT_V2 = """你是严格的技术答案评分员。判断【考生答案】是否正确回答了【问题】。

你会拿到两份参考材料：
- 【参考答案要点】：出题人写的正确要点（权威依据）
- 【检索到的资料】：系统当时提供给考生的资料（可能不完整，甚至与问题无关）

**先做覆盖率检查，再定等级**（这一步决定等级，不要跳过）：
把【参考答案要点】拆成若干「关键概念」（每个概念是一个可判断有无的知识点，
例如「code-based 成本极低」「用 versioned index 并行」）。
逐个到【考生答案】里找，判断该概念是否被表达出来（同义表述算作有）。
记 c = 被覆盖的概念数，t = 概念总数。

覆盖率规则（**按顺序判断，命中即停**）：
1. c == 0 → INCORRECT。答案与参考要点没有任何重合的实质内容。
   **典型形态：答案讲的内容虽然也在说这个话题，但都属于另一套知识点，
   参考要点里的概念一个都没出现。** 这种「听起来专业但答的是别的」必须判 INCORRECT。
   例：问「怎么用查询改写提高命中率」，答「表格解析／权限过滤／转人工」→ c==0 → INCORRECT。
2. c / t < 0.5 → INCORRECT。多数关键概念没覆盖。
3. c / t < 1 → PARTIAL。答对了主要部分，但漏了关键概念。
4. c == t → CORRECT。

其余规则：
- **不要因为措辞与参考要点不同就判错**——只要意思一致就算该概念被覆盖。
- 答案比要点更简略、少讲一个次要细节，不影响覆盖判定。
- 若答案的某条陈述与【参考答案要点】直接矛盾，无论覆盖率多少，判 INCORRECT。
- 只输出 JSON，不要解释、不要 Markdown 代码块。

输出 JSON：
{"covered": ["被覆盖的概念"], "missing": ["未覆盖的概念"],
 "grade": "CORRECT|PARTIAL|INCORRECT|REFUSED", "reason": "一句话"}
"""

ACCURACY_PROMPT_V3 = """你是严格的技术答案评分员。判断【考生答案】是否正确回答了【问题】。

你会拿到两份参考材料：
- 【参考答案要点】：出题人写的正确要点（权威依据）
- 【检索到的资料】：系统当时提供给考生的资料（可能不完整，甚至与问题无关）

评分等级（必须选一个）：
- CORRECT：答案与【参考答案要点】在实质上一致。措辞不同、更简略、少讲一个次要细节，都算 CORRECT。
- PARTIAL：答对了一部分，但遗漏关键要点或含次要错误。
- INCORRECT：**答非所问**——答案通篇讲的是另一件事，或与【参考答案要点】存在事实矛盾。
- REFUSED：答案表示资料不足／拒绝作答。

**最关键的一条：区分「答得不全」和「答的不是这个」**
- 答案覆盖了参考要点的主要方向、只是漏了某个细节 → PARTIAL。
- 答案**通篇在讲另一个知识点**，虽然语言流畅、内容本身也对，但回答的不是这个问题 → **INCORRECT**。
  判断依据：把答案逐句拿去对照【参考答案要点】，如果命中不了任何一条要点，就是 INCORRECT。
- 特别注意**近邻问题**：A 题与 B 题同属一个领域（比如都在讲 RAG）。如果考生答的是 B 题的答案，
  必须判 INCORRECT，不能因为「话题相关」就判 PARTIAL。

以下是两个示例，说明「答非所问」长什么样：

示例一（应判 INCORRECT）
【问题】混合检索里 RRF 的公式是什么？k 取多少？
【参考答案要点】score(d) = Σ_i 1/(k + rank_i(d))，k 通常取 60；只用排名不用分数
【考生答案】常用的向量相似度度量有余弦相似度、点积和欧氏距离。选择哪个取决于
embedding 模型训练时使用的度量方式，以确保语义空间的一致性。
→ 判定 INCORRECT：答案通篇讲相似度度量，本身正确，但没有一句涉及 RRF 公式或 k 值。

示例二（应判 CORRECT）
【问题】混合检索里 RRF 的公式是什么？k 取多少？
【参考答案要点】score(d) = Σ_i 1/(k + rank_i(d))，k 通常取 60；只用排名不用分数
【考生答案】就是对每路检索结果取「1 除以（常数加上它在这一路的排名）」再相加，常数一般用 60。
→ 判定 CORRECT：用白话表达、没有 Σ 符号，但语义与要点一致。

其余规则：
1. 以【参考答案要点】为准，不要用你自己的知识去补充或推翻它。
2. 不要因为措辞与参考要点不同就判错——只要意思一致就算覆盖。
3. 只输出 JSON，不要解释、不要 Markdown 代码块。

输出 JSON：
{"grade": "CORRECT|PARTIAL|INCORRECT|REFUSED", "reason": "一句话",
 "missing": "漏掉或答错的关键概念，逗号分隔；没有则空字符串"}
"""

# 三版都保留（见变更记录 2026-10-14（四）(五)）。
#   v1：在 49 条人工标注上与人工一致 42 条（85.7%），接近人工真值（69.4%），
#       但偏乐观（宽大 8 条、冤枉 0 条），且从不输出 INCORRECT。
#   v2：一致仅 24 条（49.0%），把「漏一个次要细节」当不够格，会编造 ref 里不存在的缺失项，已弃用。
#   v3：v1 + few-shot，专门教「答非所问 vs 答得不全」的区分，并用困难集验收。
PROMPTS = {"v1": ACCURACY_PROMPT_V1, "v2": ACCURACY_PROMPT_V2,
           "v3": ACCURACY_PROMPT_V3}
DEFAULT_PROMPT = "v3"
ACCURACY_PROMPT = PROMPTS[DEFAULT_PROMPT]

GRADE_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0, "REFUSED": 0.0}


def _parse_obj(raw: str) -> dict:
    """从模型输出取 JSON 对象。

    不能用 kb_agent.parse_json_lenient()——那是 RAG 答案专用校验器，
    强制要求 answerable/answer/citations/missing 四个字段。
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|```$", "", s).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("输出不是 JSON 对象")
    return obj


def _chat_json(messages: list[dict], retries: int = 2) -> dict:
    last = None
    for _ in range(retries + 1):
        try:
            return _parse_obj(K.chat(messages, temperature=0.0))
        except Exception as e:                              # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"取 JSON 失败: {last}")


def build_context(hits: list[dict]) -> str:
    lines = [f"[{i}] 来源：{h['doc']} / {h['cid']}\n{h['text']}"
             for i, h in enumerate(hits, 1)]
    return "<资料>\n" + "\n\n".join(lines) + "\n</资料>"


def _norm(s: str) -> str:
    """只保留中日韩字符与字母数字，用于宽松包含判断。

    下划线和连字符都要去掉：判据可能写 `u_k`，而参考答案里可能是
    `u_(k-1)` 或干脆写成 `uk`，保留分隔符会导致本应通过的匹配失败。
    """
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", s or "").replace("_", "").lower()


def _is_grounded(item: str, ref: str) -> bool:
    """判据报的「缺失项」是否在参考答案要点里有依据。

    为什么要校验：实测判据会**凭空编造缺失项**——例如对「R2R 控制」那道题报
    「漏掉：RRF 公式」，而 RRF 是检索融合的公式，与 run-to-run 控制毫无关系，
    ref 里也没有这一项。这类幻觉会让 `missing` 字段误导人工复核。

    判据用**字符级信息覆盖**，因为试过的两种词级做法都不行：
      - 整串包含 → 大量误报。判据写「EWMA控制器的具体作用」，ref 是
        「R2R 用 EWMA 控制器按 u_k=… 更新」，整串当然不出现，但内容全在。
      - n-gram 命中率 → 既误报又漏报。4-gram 碎片（「控制器的」「制器的具」）
        拉低比例造成误报；而「三层递进」对 ref 的「层层递进」只差一个字，
        任何 n-gram 包含都判不中，造成漏报。

    现在的做法：只看**实质字符**（去掉「作用/具体/公式」这类泛化词、
    以及标点和空白），要求这些字符有足够比例出现在 ref 里。
    措辞差异大但内容相同的缺失项会通过；凭空编造的词（如「RRF」）字符几乎全不在 ref 中。
    数字与英文串单独要求至少命中一个，因为它们是最硬的证据。
    """
    ref_n = _norm(ref)
    if not ref_n:
        return True

    generic = ["具体", "作用", "关键", "内容", "部分", "方面", "问题", "情况",
               "相关", "以及", "或者", "是否"]
    STOP_CHARS = set("的了和与及或在把被让使从到对为是这那个我你他它其之中等"
                     "更极快慢高低多少大小上下前后如果则需应可要会能该哪些种样")
    s = item or ""
    for g in generic:
        s = s.replace(g, " ")
    # 泛化词去掉后剩下的中文串
    zh = "".join(re.findall(r"[\u4e00-\u9fff]", s))
    ascii_tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_.\-/]*|\d+(?:\.\d+)?", s)]
    ascii_tokens = [t for t in dict.fromkeys(ascii_tokens) if len(t) >= 2]

    # 英文/数字是最硬的证据：只要有一个完全没出现，就判为无依据。
    # 注意要比归一化后的形式——ref 侧已经过 _norm（连字符被去掉），
    # 所以 "code-based" 要变成 "codebased" 才能和 ref 里的写法对上。
    for t in ascii_tokens:
        if _norm(t) not in ref_n:
            return False

    if not zh:
        # 只有英文/数字，且都命中了
        return bool(ascii_tokens)

    # 汉语字的能力差异太大：虚词（的/了/和/更/极…）几乎在任何 ref 里都能碰到，
    # 拿它们算覆盖率会把编造项也算成"有依据"。所以只统计内容字。
    chars = [c for c in dict.fromkeys(zh) if c not in STOP_CHARS]
    if not chars:                      # 全是虚词，无法判断，按有依据处理（宁松不严）
        return True
    hits = sum(1 for c in chars if c in ref_n)
    return hits >= 2 and hits / len(chars) >= 0.5


def judge_accuracy(q: str, answer: str, ref: str, ctx: str,
                   prompt: str | None = None) -> dict:
    obj = _chat_json([
        {"role": "system", "content": prompt or ACCURACY_PROMPT},
        {"role": "user", "content":
            f"【问题】\n{q}\n\n【参考答案要点】\n{ref}\n\n{ctx}\n\n【考生答案】\n{answer}"},
    ])
    g = str(obj.get("grade", "")).upper().strip()
    if g not in GRADE_SCORE:
        g = "INCORRECT"

    # 模型可能输出列表或字符串，两种都接。
    def as_list(v) -> list[str]:
        if isinstance(v, list):
            return [str(x)[:80] for x in v][:12]
        s = str(v or "").strip()
        return [p.strip()[:80] for p in re.split(r"[,，;；]", s) if p.strip()][:12]

    missing = as_list(obj.get("missing"))
    # 拦掉「没有/无/空」这类表示"没漏"的占位值——它们不是缺失项。
    # 实测判据会输出 missing="没有"，若不拦会被当成一个凭空编造的缺失项。
    placeholder = {"没有", "无", "空", "none", "n/a", "无缺失", "没有缺失", ""}
    missing = [m for m in missing if m.strip().lower() not in placeholder]
    # 逐条检查缺失项在 ref 里是否有依据（防判据幻觉）
    ungrounded = [m for m in missing if not _is_grounded(m, ref)]
    return {"grade": g, "score": GRADE_SCORE[g],
            "reason": str(obj.get("reason") or "")[:240],
            "missing": "；".join(missing)[:240],
            "missing_list": missing,
            "missing_ungrounded": ungrounded,
            "covered_list": as_list(obj.get("covered"))}


def eval_dir_of(kb: dict, here: Path = HERE) -> Path:
    """评测集目录：默认 kb/<id>/eval；注册表给了 eval_dir 时用它。

    dram 库复用 demo 的 38 条评测集（gold cid 与 webui 索引完全相同）。
    """
    if kb.get("eval_dir"):
        return (here / kb["eval_dir"]).resolve()
    return here / kb["docs"].replace("docs", "eval")


def kb_ids() -> list[str]:
    """注册表里的全部知识库 id（保持顺序）。"""
    reg = json.loads((HERE / "kb" / "registry.json").read_text("utf-8"))
    return [k["id"] for k in reg["kbs"]]


def run(only: str | None, ver: str = DEFAULT_PROMPT) -> None:
    prompt = PROMPTS[ver]
    reg = {k["id"]: k for k in json.loads(
        (HERE / "kb" / "registry.json").read_text("utf-8"))["kbs"]}
    all_metrics = {}
    for kb_id in kb_ids():
        if only and kb_id != only:
            continue
        rep_f = HERE / "kb" / kb_id / "eval" / "report_generation.json"
        if not rep_f.exists():
            print(f"[跳过] {kb_id}：还没有生成层报告（先跑 run_eval.py --gen）\n")
            continue
        d = json.loads(rep_f.read_text("utf-8"))
        rows = d["cases"]
        chunks = {c["cid"]: c for c in json.loads(
            (HERE / "kb" / kb_id / "index" / "chunks.json").read_text("utf-8"))}
        ev_file = eval_dir_of(reg[kb_id]) / "eval_set.jsonl"
        ev = {c["q"]: c for c in (
            json.loads(l) for l in ev_file.read_text("utf-8").splitlines()
            if l.strip() and not l.startswith("#"))}

        print("=" * 80)
        print(f"[{kb_id}] 答案正确性评分：{len(rows)} 条")

        out = []
        for i, row in enumerate(rows, 1):
            hits = [{"cid": chunks[c]["cid"], "doc": chunks[c]["doc"],
                     "text": chunks[c]["text"]} for c in row["hits"] if c in chunks]
            # 拒答题不评正确性（它没给答案），单独统计
            if not row["judged_answerable"]:
                out.append({"q": row["q"], "cat": row["cat"],
                            "expected_answerable": row["expected_answerable"],
                            "judged_answerable": False, "retrieval_hit": row["retrieval_hit"],
                            "answer": row["answer"],
                            "grade": "REFUSED", "score": None,
                            "reason": "模型拒答，按预期" if not row["expected_answerable"]
                                      else "模型拒答但该题可答",
                            "missing": "", "ref": ev[row["q"]].get("ref", "")})
                continue
            t0 = time.time()
            r = judge_accuracy(row["q"], row["answer"] or "",
                              ev[row["q"]].get("ref", ""), build_context(hits),
                              prompt=prompt)
            out.append({"q": row["q"], "cat": row["cat"],
                        "expected_answerable": row["expected_answerable"],
                        "judged_answerable": True,
                        "retrieval_hit": row["retrieval_hit"],
                        "answer": row["answer"], "hits": row["hits"],
                        "ref": ev[row["q"]].get("ref", ""), **r})
            if i % 5 == 0 or i == len(rows):
                print(f"    ... {i}/{len(rows)}  (最近 {r['grade']}, {time.time()-t0:.1f}s)")

        graded = [x for x in out if x["score"] is not None]
        cnt = {g: sum(1 for x in graded if x["grade"] == g)
               for g in ("CORRECT", "PARTIAL", "INCORRECT")}
        n = len(graded)
        # 判据 missing 字段的幻觉统计（缺失项在 ref 里找不到依据）
        ungrounded_all = [m for x in out for m in (x.get("missing_ungrounded") or [])]
        n_missing = sum(len(x.get("missing_list") or []) for x in out)
        metrics = {
            "n_cases": len(rows),
            "n_graded": n,
            "n_refused": len(out) - n,
            "correct_n": cnt["CORRECT"],
            "partial_n": cnt["PARTIAL"],
            "incorrect_n": cnt["INCORRECT"],
            "accuracy_mean": round(statistics.mean(x["score"] for x in graded), 4) if n else None,
            "correct_rate": round(cnt["CORRECT"] / n, 4) if n else None,
            # 看趋势用这个：把「答对但漏要点」也算进问题
            "wrong_or_partial_rate":
                round((cnt["PARTIAL"] + cnt["INCORRECT"]) / n, 4) if n else None,
            # missing 字段的可信度：缺失项总数 / 其中在 ref 里找不到依据的条数
            "n_missing_items": n_missing,
            "n_missing_ungrounded": len(ungrounded_all),
            "missing_grounded_rate":
                round(1 - len(ungrounded_all) / n_missing, 4) if n_missing else None,
        }
        rep_out = HERE / "kb" / kb_id / "eval" / f"report_accuracy_{ver}.json"
        rep_out.write_text(json.dumps(
            {"kb": kb_id, "prompt_version": ver, "metrics": metrics, "cases": out},
            ensure_ascii=False, indent=1), encoding="utf-8")
        all_metrics[kb_id] = metrics

        print(f"\n  === 答案正确性指标（提示词 {ver}）===")
        for k, v in metrics.items():
            if k != "n_cases":
                print(f"    {k:30} {v}")
        bad = [x for x in graded if x["grade"] != "CORRECT"]
        if bad:
            print(f"\n  未得满分的 {len(bad)} 条（PARTIAL = 答对但漏要点，不等于答错）：")
            for x in sorted(bad, key=lambda y: y["score"]):
                print(f"    [{x['grade']:9} {x['score']}] {x['q'][:42]}")
                print(f"        理由: {x['reason'][:96]}")
                if x["missing"]:
                    print(f"        漏掉: {x['missing'][:96]}")
                ug = x.get("missing_ungrounded") or []
                if ug:
                    print(f"        ⚠ 其中 {len(ug)} 项在 ref 里找不到依据（判据可能凭空编造）: "
                          f"{'；'.join(ug)[:80]}")
        if ungrounded_all:
            print(f"\n  ⚠ 判据 missing 幻觉：{len(ungrounded_all)}/{n_missing} 个缺失项"
                  f"在 ref 里无依据（grounded_rate={metrics['missing_grounded_rate']}）")
            print(f"     这些项不能当事实引用，只能当线索。")
        print(f"\n  报告已写入 kb/{kb_id}/eval/report_accuracy_{ver}.json\n")

    if len(all_metrics) > 1:
        print("=" * 80)
        print(f"汇总（提示词 {ver}）")
        keys = ["n_graded", "correct_n", "partial_n", "incorrect_n",
                "correct_rate", "accuracy_mean", "wrong_or_partial_rate"]
        print(f"  {'库':<12}" + "".join(f"{k[:16]:>18}" for k in keys))
        for kb_id, m in all_metrics.items():
            print(f"  {kb_id:<12}" + "".join(f"{str(m.get(k)):>18}" for k in keys))
    print("\n提示：判据能力已用困难集验收（见 judge_hard_eval.py）：")
    print("      v3 对远域答非所问的召回 1.00、对白话正确答案的保持 1.00。")
    print("      但在真实数据的 49 条人工标注上 Kappa 仅 0.516（偏乐观，宽大 7 条），")
    print("      所以引用正确性数字时必须带偏差说明，不要直接给裸数字。见变更记录。")


def main() -> int:
    ap = argparse.ArgumentParser(description="答案正确性评测")
    ap.add_argument("--only", default=None)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS),
                    help="判据提示词版本：v1 偏乐观、v2 偏悲观（默认 v2）")
    args = ap.parse_args()
    run(args.only, args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
