# -*- coding: utf-8 -*-
"""标注者间一致性（IAA）分析。

回答三个问题
------------
1. **两个人类标注者之间有多一致？**（这是关键上限）
   如果两个人彼此都不太一致，说明「答对/答对但漏要点/答错」这条边界本身模糊，
   那么任何 LLM 判据的 Kappa 都有一个达不到 0.7 的天花板。
2. **哪些题最容易分歧？** 逐条列出分歧项，这才是修订评分标准（rubric）的依据。
3. **判据和人类差多少？** 把 LLM 判据的预测也拉进来做三方对比。

注意：判据的预测需要跑 LLM（加 --with-judge）。不加则只比较人类之间。

用法：
  python iaa_calc.py --annotators alice bob
  python iaa_calc.py --annotators alice bob --with-judge        # 三方对比（花额度）
  python iaa_calc.py --annotators alice bob --detail            # 打印每条分歧
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
sys.path.insert(0, str(HERE))

ORDER = ["INCORRECT", "PARTIAL", "CORRECT"]
SAMPLE_F = KB / "_annotation_sample.json"
HELPER_F = KB / "_annotation_helpers.json"


def load_labels(name: str) -> dict[str, str]:
    f = KB / f"_annotations_{name}.json"
    if not f.exists():
        raise SystemExit(f"找不到标注文件：{f}")
    return json.loads(f.read_text("utf-8")).get("labels", {})


def kappa(pairs: list[tuple[str, str]], weighted: bool = False):
    """直接算即可，复用 judge_calibration 里已自检过的实现。"""
    import judge_calibration as JC
    return JC.cohen_kappa(pairs, ORDER, weights="linear" if weighted else None)


def confusion(pairs: list[tuple[str, str]]) -> dict:
    m = {a: {b: 0 for b in ORDER} for a in ORDER}
    for a, b in pairs:
        m[a][b] += 1
    return m


def print_confusion(title: str, pairs: list[tuple[str, str]],
                    row_name: str, col_name: str) -> None:
    print(f"\n  {title}")
    print(f"    {'':<12}" + "".join(f"{c:>11}" for c in ORDER) + f"{'合计':>8}")
    m = confusion(pairs)
    for r in ORDER:
        print(f"    {row_name}={r:<10}" if len(r) < 4 else f"    {r:<12}"
              + "".join(f"{m[r][c]:>11}" for c in ORDER) + f"{sum(m[r].values()):>8}")
    print(f"    {'合计':<12}" + "".join(
        f"{sum(m[r][c] for r in ORDER):>11}" for c in ORDER) + f"{len(pairs):>8}")


def verdict(k: float) -> str:
    if k < 0.20:
        return "差"
    if k < 0.40:
        return "一般"
    if k < 0.60:
        return "中等"
    if k < 0.80:
        return "好"
    return "很好"


def judge_predictions(sample: list[dict]) -> dict[str, str]:
    """用当前默认提示词跑判据，返回 {id: grade}。会消耗额度。"""
    import accuracy_eval as AE
    helpers = json.loads(HELPER_F.read_text("utf-8"))
    gen_h = helpers.get("gen_helpers", {})

    # 各库的 chunk 正文（判据需要它才能核对答案是否有据）
    chunk_text: dict[str, dict[str, dict]] = {}
    for kb_id in ("dram", "ai_prep", "agent_eval"):
        f = KB / kb_id / "index" / "chunks.json"
        if f.exists():
            chunk_text[kb_id] = {c["cid"]: c for c in
                                 json.loads(f.read_text("utf-8"))}

    preds: dict[str, str] = {}
    for c in sample:
        ref, hits = "", []
        key = f"gen:{c['kb']}\x00{c['q']}"
        if key in gen_h:
            ref = gen_h[key].get("ref", "") or c.get("ref", "")
            hits = gen_h[key].get("hits") or []
        else:
            ref = c.get("ref", "")
        pool = chunk_text.get(c["kb"], {})
        ctx_hits = [pool[h] for h in hits if h in pool]
        try:
            r = AE.judge_accuracy(c["q"], c["answer"], ref,
                                  AE.build_context(ctx_hits))
            preds[c["id"]] = r["grade"]
        except Exception as e:                              # noqa: BLE001
            print(f"  !! 判据失败 {c['id']}: {e}")
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description="标注者间一致性分析")
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--with-judge", action="store_true", help="加入 LLM 判据做三方对比（花额度）")
    ap.add_argument("--detail", action="store_true", help="打印逐条分歧")
    args = ap.parse_args()

    sample = json.loads(SAMPLE_F.read_text("utf-8"))["cases"]
    by_id = {c["id"]: c for c in sample}
    labels = {a: load_labels(a) for a in args.annotators}

    for a, l in labels.items():
        print(f"标注者 {a}：{len(l)} 条")

    # 只用所有人都标过的题
    # 注意：这里不能用 `all(x in labels[a] for a in annotators)` 且外层循环变量也叫 cid，
    # Python 会把推导式里的 cid 当成自由变量而报 NameError。用不同名字避免。
    names = list(args.annotators)
    common = [c["id"] for c in sample
              if all(c["id"] in labels[n] for n in names)]
    print(f"\n共同标注的题：{len(common)} 条")
    if len(common) < 10:
        print("⚠ 共同标注太少（<10），一致性估计不可靠。")
        return 1

    report: dict = {"n_common": len(common), "pairs": {}, "judge": None}

    # ---- 两两人类标注者 ----
    print("\n" + "=" * 84)
    print("① 人类标注者之间的一致性（这是 Kappa 的上限参考）")
    for a, b in itertools.combinations(args.annotators, 2):
        pairs = [(labels[a][cid], labels[b][cid]) for cid in common]
        po, k = kappa(pairs)
        _, kw = kappa(pairs, weighted=True)
        print(f"\n  {a} vs {b}")
        print(f"    原始一致率 {po:.3f} | 未加权 Kappa {k:.3f}（{verdict(k)}）"
              f" | 线性加权 Kappa {kw:.3f}")
        print_confusion("混淆矩阵（行=" + a + "，列=" + b + "）", pairs, a, b)
        report["pairs"][f"{a}|{b}"] = {"agreement": round(po, 4),
                                       "kappa": round(k, 4),
                                       "kappa_weighted": round(kw, 4)}

    # ---- 判据 vs 人类 ----
    if args.with_judge:
        print("\n" + "=" * 84)
        print("② LLM 判据 vs 人类标注（花额度，需等一会）")
        preds = judge_predictions(sample)
        report["judge"] = {"n": len(preds), "vs": {}}
        for a in args.annotators:
            ids = [cid for cid in common if cid in preds]
            if not ids:
                continue
            pairs = [(labels[a][cid], preds[cid]) for cid in ids]
            po, k = kappa(pairs)
            print(f"\n  判据 vs {a}（{len(ids)} 条）")
            print(f"    原始一致率 {po:.3f} | 未加权 Kappa {k:.3f}（{verdict(k)}）")
            lenient = sum(1 for h, g in pairs if g == "CORRECT" and h != "CORRECT")
            strict = sum(1 for h, g in pairs if g != "CORRECT" and h == "CORRECT")
            print(f"    宽大（判据说对、人说不对）{lenient} 条 | "
                  f"冤枉（判据说不对、人说对）{strict} 条")
            print(f"    判据等级分布 {dict(Counter(p for _, p in pairs))}")
            report["judge"]["vs"][a] = {"n": len(ids), "agreement": round(po, 4),
                                        "kappa": round(k, 4),
                                        "lenient": lenient, "strict": strict}

    # ---- 分歧明细：这才是修订 rubric 的依据 ----
    print("\n" + "=" * 84)
    print("③ 人类分歧的题（这些题说明标准本身有歧义）")
    disputed = []
    for cid in common:
        gs = {labels[a][cid] for a in args.annotators}
        if len(gs) > 1:
            disputed.append((cid, {a: labels[a][cid] for a in args.annotators}))
    print(f"  共 {len(disputed)} / {len(common)} 条存在分歧"
          f"（{len(disputed)/len(common):.1%}）")
    for cid, gs in disputed[:40]:
        c = by_id[cid]
        print(f"\n  [{cid}] {c['source']}  " + "  ".join(f"{a}={g}" for a, g in gs.items()))
        print(f"     问题: {c['q'][:60]}")
        print(f"     要点: {(c.get('ref') or '（构造样本）')[:70]}")
        print(f"     答案: {c['answer'][:80]}")
    if not args.detail and disputed:
        print("\n  （加 --detail 可看全部分歧项）")

    # ---- 分组看：哪个来源的分歧最多 ----
    print("\n" + "=" * 84)
    print("④ 按来源分组的同意程度（哪类样本最难标）")
    groups: dict[str, list[int]] = {}
    for cid in common:
        grp = by_id[cid]["source"].split(":")[0]
        groups.setdefault(grp, []).append(1 if len({labels[a][cid] for a in args.annotators}) == 1 else 0)
    print(f"  {'来源':<10}{'样本数':>8}{'全一致率':>10}")
    for grp, vals in sorted(groups.items()):
        print(f"  {grp:<10}{len(vals):>8}{sum(vals)/len(vals):>10.1%}")

    out = KB / "_iaa_report.json"
    out.write_text(json.dumps({
        "annotators": args.annotators, "n_common": len(common),
        **report,
        "disputed": [{"id": cid, "labels": gs,
                      "q": by_id[cid]["q"], "source": by_id[cid]["source"]}
                     for cid, gs in disputed],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  报告已写入 {out.relative_to(HERE)}")

    # ---- 结论 ----
    print("\n" + "=" * 84)
    print("结论")
    if len(common) < 30:
        print(f"  ⚠ 共同标注只有 {len(common)} 条，Kappa 的置信区间很宽，"
              f"以下结论只能当方向参考。")
        print(f"    建议至少 50 条，最好 100+ 条。")
    pair_ks = [v["kappa"] for v in report["pairs"].values()]
    if pair_ks:
        mn = min(pair_ks)
        print(f"  人类标注者之间的最低 Kappa = {mn:.3f}")
        if mn < 0.6:
            print(f"  ⚠ 人类之间尚未达到 0.6，说明评分标准的边界本身需要先收紧"
                  f"（尤其「答对但漏要点」vs「答对」）。")
            print(f"    在此之前，不应把判据 Kappa < 0.7 当作判据的缺陷——"
                  f"人自己都做不到的事，判据做不到是正常的。")
            print(f"    具体怎么收紧：看上面 ③ 的分歧项，把「哪些算关键要点」写进 rubric。")
        else:
            print(f"  ✅ 人类之间一致性可接受，此时判据与人类的差距才是需要改进的部分。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
