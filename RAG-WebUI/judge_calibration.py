# -*- coding: utf-8 -*-
"""判据校准：比较人工标注与 LLM 判据的一致率、Cohen's Kappa、加权 Kappa。

为什么要校准
------------
`accuracy_eval.py` 的所有结论都建立在一个**未经校准**的 LLM 判据上。
如果判据本身不准，`correct_rate` / `wrong_or_partial_rate` 这些数字都得打折。
这个脚本就是去量化「判据到底有多准」。

方法
----
人工标注在**查看判据结论之前**独立完成（规则写在 judge_calibration_labels.json 的
rubric 里，可复现、可复核、可修改）。然后：

  - 原始一致率（accuracy）
  - Cohen's Kappa（未加权）：κ = (p_o − p_e) / (1 − p_e)
  - **加权 Kappa**（线性权重）：等级是有序的（INCORRECT < PARTIAL < CORRECT），
    把「差一级」和「差两级」同等看待会低估判据——所以加权 Kappa 更合适，
    **结论以它为准**。
  - 混淆矩阵：看错误集中在哪个方向
  - **两个方向的不对称错误**（这才是真正影响结论的）：
      * 漏放：LLM 给 CORRECT 但人工认为是错的 → 会让指标虚高，最危险
      * 冤枉：LLM 给 INCORRECT 但人工认为答对了 → 会让指标虚低

Cohen's Kappa 的解读惯例（Landis & Koch）：
  <0.20 差 / 0.21–0.40 一般 / 0.41–0.60 中等 / 0.61–0.80 好 / 0.81–1.00 很好

用法：
  python judge_calibration.py                # 跑校准
  python judge_calibration.py --rerun        # 重新调判据打分（默认复用 report_accuracy.json）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "RAG知识库Agent" / "demo"
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(HERE))

LABELS = HERE / "judge_calibration_labels.json"


def cohen_kappa(pairs: list[tuple[str, str]], order: list[str],
                weights: str | None = None) -> tuple[float, float]:
    """返回 (原始一致率, kappa)。weights=None 为未加权，'linear' 为线性加权。

    统一按「分歧」写，避免符号搞错：
        未加权：κ = 1 − D_o / D_e，其中 D_o = 1 − p_o，D_e = 1 − p_e
        线性加权：w_ij = 1 − |i−j|/(K−1)
                 κ_w = 1 − Σ w_ij·O_ij / Σ w_ij·E_ij
        （等价写法见 Cohen 1968；`po − pe` 与 `De − Do` 恒等。）

    注意：这里踩过一次坑——最初把公式写成 (p_o − p_e)/(1 − p_e)，
    在加权分支里却按 ΣwO/ΣwE 求比值，两者方向相反，
    结果算出 κ_w = −0.321，而同一批数据的未加权 κ 是 0.698、原始一致率 0.833。
    加权 Kappa 不可能落在未加权 Kappa 的反方向，所以那个值必然是错的。
    """
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    idx = {g: i for i, g in enumerate(order)}
    K = len(order)

    O = [[0] * K for _ in range(K)]
    for a, b in pairs:
        O[idx[a]][idx[b]] += 1
    row = [sum(O[i]) for i in range(K)]
    col = [sum(O[i][j] for i in range(K)) for j in range(K)]

    po = sum(O[i][i] for i in range(K)) / n

    if weights is None:
        pe = sum(row[i] * col[i] for i in range(K)) / (n * n)
        if abs(1 - pe) < 1e-12:
            return po, 1.0 if po == 1 else 0.0
        do, de = 1 - po, 1 - pe
        return po, 1 - do / de

    # 线性加权
    def w(i, j):
        return 1 - abs(i - j) / (K - 1) if K > 1 else 1.0

    do = sum(w(i, j) * O[i][j] for i in range(K) for j in range(K)) / n
    de = sum(w(i, j) * row[i] * col[j] for i in range(K) for j in range(K)) / (n * n)
    if abs(de) < 1e-12:
        return po, 1.0
    return po, 1 - do / de


def self_test() -> bool:
    """用已知答案验证 Kappa 实现。数值取自 Cohen (1968) 的经典算例。

    这是必要的：公式写反过一次，而错误结果（κ=−0.321）看起来像个"结论"，
    如果不去质疑它，就会得出"判据不可用"的错误判断。
    """
    order = ["a", "b", "c"]
    # Cohen 1968 经典 2x2 例子（等效 3 类）：agree 30/50, pe=0.5 -> kappa=0.2
    pairs = ([("a", "a")] * 20 + [("b", "b")] * 15 + [("a", "b")] * 5
             + [("b", "a")] * 10)
    po, k = cohen_kappa(pairs, ["a", "b"])
    ok1 = abs(po - 0.70) < 1e-9 and abs(k - 0.40) < 1e-9

    # 完全一致 -> kappa = 1
    _, k2 = cohen_kappa([("a", "a")] * 5, order)
    ok2 = abs(k2 - 1.0) < 1e-9

    # 完全随机（行列边际独立）-> kappa ≈ 0
    rnd = ([("a", "a")] * 25 + [("a", "b")] * 25 + [("b", "a")] * 25
           + [("b", "b")] * 25)
    _, k3 = cohen_kappa(rnd, ["a", "b"])
    ok3 = abs(k3) < 1e-9

    # 加权：相邻分歧应比远处分歧罚得轻 -> kappa_w > kappa
    near = [("a", "a")] * 8 + [("b", "b")] * 8 + [("a", "b")] * 2
    far = [("a", "a")] * 8 + [("b", "b")] * 8 + [("a", "c")] * 2
    _, k_near = cohen_kappa(near, order, weights="linear")
    _, k_far = cohen_kappa(far, order, weights="linear")
    ok4 = k_near > k_far

    ok = ok1 and ok2 and ok3 and ok4
    print("Kappa 实现自检：")
    print(f"  Cohen 算例      po={po:.3f} kappa={k:.3f}  {'OK' if ok1 else 'FAIL'}（应为 0.700 / 0.400）")
    print(f"  完全一致        kappa={k2:.3f}  {'OK' if ok2 else 'FAIL'}（应为 1.000）")
    print(f"  边际独立        kappa={k3:.3f}  {'OK' if ok3 else 'FAIL'}（应为 0.000）")
    print(f"  加权方向        近邻分歧 {k_near:.3f} > 远距分歧 {k_far:.3f}  "
          f"{'OK' if ok4 else 'FAIL'}")
    print(f"  => {'自检通过' if ok else '自检失败，公式仍有问题'}\n")
    return ok


def verdict_of(k: float) -> str:
    if k < 0.20:
        return "差"
    if k < 0.40:
        return "一般"
    if k < 0.60:
        return "中等"
    if k < 0.80:
        return "好"
    return "很好"


def main() -> int:
    ap = argparse.ArgumentParser(description="判据校准")
    ap.add_argument("--rerun", action="store_true",
                    help="重新调判据打分（默认复用 report_accuracy_<ver>.json，不花额度）")
    ap.add_argument("--prompt", default=AE.DEFAULT_PROMPT,
                    choices=sorted(AE.PROMPTS),
                    help=f"判据提示词版本（默认 {AE.DEFAULT_PROMPT}）")
    args = ap.parse_args()

    cfg = json.loads(LABELS.read_text("utf-8"))
    order = cfg["grades_order"]
    labels = cfg["labels"]

    if not self_test():
        print("公式自检未通过，先修实现再谈校准结论。")
        return 1

    # 取判据结论
    if args.rerun:
        import accuracy_eval as AE
        import run_eval as RE
        preds = {}
        for kb_id in ("ai_prep", "agent_eval"):
            gen = {x["q"]: x for x in json.loads(
                (HERE / "kb" / kb_id / "eval" / "report_generation.json")
                .read_text("utf-8"))["cases"]}
            chunks = {c["cid"]: c for c in json.loads(
                (HERE / "kb" / kb_id / "index" / "chunks.json").read_text("utf-8"))}
            ev = {c["q"]: c for c in (
                json.loads(l) for l in (HERE / "kb" / kb_id / "eval" / "eval_set.jsonl")
                .read_text("utf-8").splitlines() if l.strip() and not l.startswith("#"))}
            for item in labels:
                if item["kb"] != kb_id:
                    continue
                row = gen[item["q"]]
                hits = [{"cid": chunks[c]["cid"], "doc": chunks[c]["doc"],
                         "text": chunks[c]["text"]} for c in row["hits"] if c in chunks]
                r = AE.judge_accuracy(row["q"], row["answer"] or "",
                                      ev[row["q"]].get("ref", ""),
                                      AE.build_context(hits),
                                      prompt=AE.PROMPTS[args.prompt])
                preds[(kb_id, row["q"])] = r["grade"]
        print(f"（已用提示词 {args.prompt} 重新打分 {len(preds)} 条）\n")
    else:
        preds = {}
        for kb_id in ("ai_prep", "agent_eval"):
            # 报告按提示词版本分开存；优先读所请求版本，回退到旧文件名
            for name in (f"report_accuracy_{args.prompt}.json", "report_accuracy.json"):
                f = HERE / "kb" / kb_id / "eval" / name
                if f.exists():
                    break
            for x in json.loads(f.read_text("utf-8"))["cases"]:
                if x.get("score") is not None:
                    preds[(kb_id, x["q"])] = x["grade"]
        print(f"（复用 kb/<id>/eval/{f.name}）\n")

    rows = []
    for item in labels:
        key = (item["kb"], item["q"])
        if key not in preds:
            print(f"  !! 判据结果里找不到：{item['q'][:40]}（该题可能被拒答）")
            continue
        rows.append((item, preds[key]))

    print("=" * 92)
    print("逐条对照（人工标注 vs LLM 判据）")
    print(f"  {'人工':<10}{'判据':<10}问题")
    agree = 0
    for item, p in rows:
        mark = "  " if item["human"] == p else "≠ "
        if item["human"] == p:
            agree += 1
        print(f"  {mark}{item['human']:<8}{p:<10}{item['q'][:44]}")
        if item["human"] != p:
            print(f"        人工理由: {item['human_reason'][:84]}")
    print(f"\n  一致 {agree}/{len(rows)}")

    pairs = [(item["human"], p) for item, p in rows]
    po, k = cohen_kappa(pairs, order)
    _, kw = cohen_kappa(pairs, order, weights="linear")

    # 类别覆盖检查：某一类在两个评分者里都/任一个完全缺失时，Kappa 会失效
    # （边际极不平衡 -> 期望分歧趋小 -> 未加权与加权 Kappa 都可能给出无意义的值）。
    # 实测遇到过：判据一次 INCORRECT 都没给出，导致加权 Kappa 算出 −0.321，
    # 而同一批数据未加权 Kappa 是 0.698 —— 这种矛盾是"类别缺失"的信号，不是判据方向的信号。
    human_cnt = Counter(h for h, _ in pairs)
    judge_cnt = Counter(p for _, p in pairs)
    empty = [g for g in order if human_cnt.get(g, 0) == 0 or judge_cnt.get(g, 0) == 0]

    # 严重错误：判据把人工认为错的判成 CORRECT（差两级），会让指标虚高
    mild = [(i, p) for i, p in rows
            if p == "PARTIAL" and i["human"] == "INCORRECT"]

    # 混淆矩阵
    print("\n" + "=" * 92)
    print("混淆矩阵（行=人工，列=判据）")
    print(f"  {'人工\\判据':<12}" + "".join(f"{g:>10}" for g in order) + f"{'合计':>8}")
    for g in order:
        cnt = Counter(p for h, p in pairs if h == g)
        print(f"  {g:<12}" + "".join(f"{cnt.get(x, 0):>10}" for x in order)
              + f"{sum(cnt.values()):>8}")
    print(f"  {'合计':<12}" + "".join(
        f"{sum(1 for h, p in pairs if p == x):>10}" for x in order) + f"{len(pairs):>8}")

    # 两个方向的不对称错误
    lenient = [(i, p) for i, p in rows if p == "CORRECT" and i["human"] != "CORRECT"]
    strict = [(i, p) for i, p in rows
              if p in ("INCORRECT", "REFUSED") and i["human"] == "CORRECT"]

    print("\n" + "=" * 92)
    print("校准结论")
    print(f"  原始一致率            {po:.3f}  ({agree}/{len(rows)})")
    print(f"  Cohen's Kappa（未加权）{k:.3f}  → {verdict_of(k)}")
    if empty:
        print(f"  线性加权 Kappa         {kw:.3f}  **不可用**：等级 "
              f"{'/'.join(empty)} 在人工或判据里一次都没出现，")
        print(f"                       Kappa 在该情形下会给出无意义的值"
              f"（同一批数据未加权 κ={k:.3f} 却为负，本身就是信号）。")
        print(f"                       结论以未加权 Kappa 为准。")
    else:
        print(f"  **线性加权 Kappa**     {kw:.3f}  → {verdict_of(kw)}   ← 等级有序，结论以这个为准")

    print(f"\n  判据的等级分布：{dict(judge_cnt)}")
    print(f"  人工的等级分布：{dict(human_cnt)}")
    if judge_cnt.get("INCORRECT", 0) == 0 and human_cnt.get("INCORRECT", 0) > 0:
        print(f"  ⚠ 判据**一次 INCORRECT 都没给出**，而人工标了 "
              f"{human_cnt['INCORRECT']} 条 —— 这是系统性的宽大倾向，")
        print(f"    对 eval 集的合理设计可能是对的（gold 就在库里），")
        print(f"    但它意味着正确性指标会偏乐观。")

    print(f"\n  严重错误（判据 CORRECT 但人工认为错）："
          f"{len(lenient)} 条  ← 会让指标虚高")
    for i, p in lenient:
        print(f"      {i['q'][:46]}")
        print(f"        人工: {i['human']} — {i['human_reason'][:76]}")
    print(f"  轻微错误（判据 PARTIAL 但人工认为错）：{len(mild)} 条"
          f"  ← 只差一级，影响有限")
    print(f"  冤枉（判据 INCORRECT/REFUSED 但人工认为对）：{len(strict)} 条"
          f"  ← 会让指标虚低")
    for i, p in strict:
        print(f"      {i['q'][:46]}")

    print(f"\n  判据可用性判断：")
    if len(rows) < 50 or empty:
        print(f"    ⚠ 样本 {len(rows)} 条"
              + (f"、且等级 {'/'.join(empty)} 缺失" if empty else "")
              + "，**不足以给出可靠的校准结论**。")
        print(f"    当前能说的：严重错误 {len(lenient)} 条、冤枉 {len(strict)} 条，"
              f"原始一致率 {po:.0%}。")
        print(f"    要在正式引用数字前使用，需要扩到 50+ 条人工标注，且各等级都有样本。")
    elif k >= 0.8:
        print(f"    ✅ 未加权 Kappa {k:.3f}（很好）：判据可用于正式结论。")
    elif k >= 0.6:
        print(f"    ⚠ 未加权 Kappa {k:.3f}（好）：判据大体可用，引用数字时应说明分歧程度。")
    else:
        print(f"    ❌ 未加权 Kappa {k:.3f} 偏低：判据需先改提示词，再重跑校准。")

    out = HERE / "kb" / "_judge_calibration.json"
    out.write_text(json.dumps({
        "n": len(rows), "agreement": round(po, 4),
        "kappa_unweighted": round(k, 4),
        "kappa_linear_weighted": None if empty else round(kw, 4),
        "kappa_weighted_unusable_reason":
            f"等级 {'/'.join(empty)} 在人工或判据中缺失" if empty else None,
        "judge_grade_distribution": dict(judge_cnt),
        "human_grade_distribution": dict(human_cnt),
        "lenient_errors": [i["q"] for i, _ in lenient],
        "mild_errors": [i["q"] for i, _ in mild],
        "strict_errors": [i["q"] for i, _ in strict],
        "per_case": [{"kb": i["kb"], "q": i["q"], "human": i["human"],
                      "judge": p, "human_reason": i["human_reason"]}
                     for i, p in rows],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  报告已写入 kb/_judge_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
