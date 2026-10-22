# -*- coding: utf-8 -*-
"""用「困难集」验收判据，并同时在原 49 条标注上验证，防止过拟合。

困难集（kb/_hard_suite.json）：10 条答非所问（INCORRECT）+ 10 条白话正确（CORRECT）。
  它测的是判据的**判别力**：能不能识别答非所问，同时不冤枉措辞不同的正确答案。

原 49 条标注（judge_calibration_labels.json）：真实作答，CORRECT/ PARTIAL/INCORRECT 混合。
  它测的是判据在**真实分布**上的表现（一致性、Kappa）。

两个一起看是因为：只优化困难集容易把判据调成「见难就判错」，
在原标注上的冤枉数会立刻暴露这一点。

用法：
  python judge_hard_eval.py                    # 跑全部提示词版本
  python judge_hard_eval.py --prompt v3        # 只跑指定版本
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "RAG知识库Agent" / "demo"
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(HERE))

import accuracy_eval as AE   # noqa: E402
import judge_calibration as JC  # noqa: E402

KB = HERE / "kb"
ORDER = ["INCORRECT", "PARTIAL", "CORRECT"]


def load_data() -> dict[str, tuple]:
    out = {}
    for kb_id in ("ai_prep", "agent_eval"):
        gen = {x["q"]: x for x in json.loads(
            (KB / kb_id / "eval" / "report_generation.json").read_text("utf-8"))["cases"]}
        chunks = {c["cid"]: c for c in json.loads(
            (KB / kb_id / "index" / "chunks.json").read_text("utf-8"))}
        ev = {c["q"]: c for c in (
            json.loads(l) for l in (KB / kb_id / "eval" / "eval_set.jsonl")
            .read_text("utf-8").splitlines() if l.strip() and not l.startswith("#"))}
        out[kb_id] = (gen, chunks, ev)
    return out


def judge_one(kb_id, q, answer, prompt, data) -> str:
    gen, chunks, ev = data[kb_id]
    row = gen.get(q, {})
    hit_cids = row.get("hits") or []
    hits = [{"cid": chunks[c]["cid"], "doc": chunks[c]["doc"], "text": chunks[c]["text"]}
            for c in hit_cids if c in chunks]
    ref = ev.get(q, {}).get("ref", "")
    r = AE.judge_accuracy(q, answer, ref, AE.build_context(hits), prompt=prompt)
    return r["grade"]


def run_hard(prompt: str, data) -> dict:
    suite = json.loads((KB / "_hard_suite.json").read_text("utf-8"))["cases"]
    detail = []
    for c in suite:
        g = judge_one(c["kb"], c["q"], c["answer"], prompt, data)
        detail.append((c["group"], c["human"], g, c["q"], c["construction"]))

    def rate(group: str, human: str, want: str) -> tuple[float, int, int]:
        grp = [(h, g) for gg, h, g, _, _ in detail if gg == group]
        n = sum(1 for h, _ in grp if h == human)
        hit = sum(1 for h, g in grp if h == human and g == want)
        return (hit / n if n else 0.0), hit, n

    # 严格指标只用 far（远域错配）：判据应当识别出来
    r_inc, hit_inc, n_inc = rate("far", "INCORRECT", "INCORRECT")
    # 白话正确：不应被冤枉
    r_cor, hit_cor, n_cor = rate("paraphrase", "CORRECT", "CORRECT")
    # 邻域错配：边界情形，只作为参考，不计入达标条件
    r_near, hit_near, n_near = rate("near", "INCORRECT", "INCORRECT")

    all_pairs = [(h, g) for _, h, g, _, _ in detail]
    po, k = JC.cohen_kappa(all_pairs, ORDER)
    return {"recall_incorrect": r_inc, "hit_incorrect": (hit_inc, n_inc),
            "keep_correct": r_cor, "hit_correct": (hit_cor, n_cor),
            "near_incorrect": r_near, "hit_near": (hit_near, n_near),
            "agreement": po, "kappa": k,
            "n_judged_incorrect": sum(1 for _, _, g, _, _ in detail if g == "INCORRECT"),
            "detail": detail}


def run_real(prompt: str, data) -> dict:
    labels = json.loads((HERE / "judge_calibration_labels.json")
                        .read_text("utf-8"))["labels"]
    pairs = []
    for x in labels:
        gen = data[x["kb"]][0]
        ans = gen.get(x["q"], {}).get("answer") or ""
        pairs.append((x["human"], judge_one(x["kb"], x["q"], ans, prompt, data)))
    po, k = JC.cohen_kappa(pairs, ORDER)
    lenient = sum(1 for h, g in pairs if g == "CORRECT" and h != "CORRECT")
    strict = sum(1 for h, g in pairs if g in ("INCORRECT", "REFUSED") and h == "CORRECT")
    return {"agreement": po, "kappa": k, "lenient": lenient, "strict": strict,
            "n_judged_incorrect": sum(1 for _, g in pairs if g == "INCORRECT")}


def main() -> int:
    ap = argparse.ArgumentParser(description="判据困难集验收")
    ap.add_argument("--prompt", default=None, choices=sorted(AE.PROMPTS))
    ap.add_argument("--verbose", action="store_true", help="打印困难集逐条判定")
    args = ap.parse_args()

    data = load_data()
    versions = [args.prompt] if args.prompt else sorted(AE.PROMPTS)

    print("=" * 96)
    print("困难集验收：far=远域错配（必须识别）| near=邻域错配（边界，仅参考）| paraphrase=白话正确（不得冤枉）")
    print(f"  {'版本':<6}{'far 召回':>10}{'near 召回':>11}{'白话保持':>10}"
          f"{'全集一致':>10}{'全集Kappa':>11}{'判INCORRECT':>13}")
    hard_res = {}
    for v in versions:
        r = run_hard(AE.PROMPTS[v], data)
        hard_res[v] = r
        print(f"  {v:<6}{r['recall_incorrect']:>10.3f}{r['near_incorrect']:>11.3f}"
              f"{r['keep_correct']:>10.3f}{r['agreement']:>10.3f}{r['kappa']:>11.3f}"
              f"{r['n_judged_incorrect']:>13}")

    print("\n原 49 条真实标注（防止过拟合的关键对照）")
    print(f"  {'版本':<6}{'一致率':>10}{'Kappa':>10}{'宽大':>8}{'冤枉':>8}"
          f"{'判INCORRECT次数':>16}")
    real_res = {}
    for v in versions:
        r = run_real(AE.PROMPTS[v], data)
        real_res[v] = r
        print(f"  {v:<6}{r['agreement']:>10.3f}{r['kappa']:>10.3f}"
              f"{r['lenient']:>8}{r['strict']:>8}{r['n_judged_incorrect']:>16}")

    if args.verbose and args.prompt:
        print("\n" + "=" * 96)
        print(f"困难集逐条（{args.prompt}）")
        for grp, h, g, q, how in hard_res[args.prompt]["detail"]:
            mark = "  " if h == g else "≠ "
            print(f"  {mark}[{grp:<10}] 人工={h:<10}判据={g:<10}{q[:34]}")
            print(f"        {how[:76]}")

    # 综合判断
    print("\n" + "=" * 92)
    print("综合判断（候选：困难集召回高 且 原标注冤枉=0 且 原标注一致率不显著下降）")
    base = real_res.get("v1", {})
    for v in versions:
        h, r = hard_res[v], real_res[v]
        ok = (h["recall_incorrect"] >= 0.7 and r["strict"] == 0
              and r["agreement"] >= (base.get("agreement", 0) - 0.05))
        print(f"  {v}: 答非所问召回 {h['recall_incorrect']:.2f} | 白话保持 "
              f"{h['keep_correct']:.2f} | 真实一致率 {r['agreement']:.2f} | "
              f"冤枉 {r['strict']} → {'✅ 可用' if ok else '未达标'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
