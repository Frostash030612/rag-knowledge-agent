# -*- coding: utf-8 -*-
"""延迟重复实验：解决「单次 P95 被一条 API 长尾带偏」的问题。

问题背景（见变更记录 2026-10-14（六）第 4 节）
---------------------------------------------
dram 单次评测的延迟 P95 = 30.65s、最大 53.15s，而 ai_prep / agent_eval 只有 9.35s / 9.81s。
逐层排查后确认：不是答案长、不是并发限流、不是重试放大，而是 **API 侧偶发的生成阶段长尾**
（ttft 正常 5–8s，但首字后耗 22–46s，答案仅 80–163 字；且每次重跑卡住的是不同的题）。

统计上的要害：n=38 时 P95 恰好是「第 2 差的样本」，**一条离群值就能把 P95 翻三倍**。
所以「哪个库更慢」这个结论在单次 P95 上不可信。

本脚本的做法
------------
每个库独立跑 R 次（默认 3），每次都是完整一遍（走 ask_stream，与真实链路一致），
然后看两个问题：
  1. 中位数在多次运行之间稳定吗？（若是 → 中位数可信）
  2. P95 / 最大值在多次运行之间抖多少？（若剧烈 → 单次 P95 不可用作横向比较）

同时记录每次的「最慢那条是哪道题」，验证「长尾落在随机题目上」这个假设。

用法：
  python latency_repeat.py                 # 3 个库 × 3 次
  python latency_repeat.py --runs 5
  python latency_repeat.py --only dram --runs 3
  python latency_repeat.py --no-gen        # 只重测检索耗时（不调生成，免费）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "RAG知识库Agent" / "demo"
sys.path.insert(0, str(DEMO))
import kb_agent as K  # noqa: E402

TOPK, RERANK, POOL, ALPHA = 3, "max", 40, 0.7


def eval_dir_of(kb: dict) -> Path:
    if kb.get("eval_dir"):
        return (HERE / kb["eval_dir"]).resolve()
    return HERE / kb["docs"].replace("docs", "eval")


def one_run(idx: Path, ev: Path, gen: bool) -> dict:
    """跑一遍评测集，返回延迟统计、最长那条，以及**全部原始样本**（用于合并统计）。"""
    retriever = K.Retriever(idx)
    cases = K.load_eval_set(ev)
    ret, ttft, total = [], [], []
    slowest = None

    for c in cases:
        if not gen:
            t0 = time.time()
            retriever.search(c["q"], top_k=TOPK, mode="hybrid", rerank=RERANK,
                             rerank_pool=POOL, alpha=ALPHA)
            ret.append(time.time() - t0)
            continue

        t0 = time.time()
        first = None
        for kind, _payload in K.ask_stream(c["q"], retriever, top_k=TOPK,
                                           mode="hybrid", rerank=RERANK,
                                           rerank_pool=POOL, alpha=ALPHA):
            if kind == "delta" and first is None:
                first = time.time() - t0
        dt = time.time() - t0
        total.append(dt)
        if first is not None:
            ttft.append(first)
        if slowest is None or dt > slowest[0]:
            slowest = (round(dt, 2), c["q"][:34], round(first or 0, 2))

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {"median": round(statistics.median(xs), 2),
                "p95": round(sorted(xs)[max(0, int(len(xs) * 0.95) - 1)], 2),
                "max": round(max(xs), 2)}

    out = {"n": len(cases)}
    if gen:
        out["ttft"] = stats(ttft)
        out["total"] = stats(total)
        out["slowest"] = slowest
        out["samples_total"] = [round(x, 2) for x in total]
        out["samples_ttft"] = [round(x, 2) for x in ttft]
    else:
        out["retrieval"] = stats(ret)
        out["samples_retrieval"] = [round(x, 3) for x in ret]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="延迟重复实验")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-gen", action="store_true", help="只测检索耗时（免费）")
    args = ap.parse_args()

    reg = json.loads((HERE / "kb" / "registry.json").read_text("utf-8"))
    report: dict = {}

    for kb in reg["kbs"]:
        if args.only and kb["id"] != args.only:
            continue
        idx = HERE / kb["index"]
        ev = eval_dir_of(kb) / "eval_set.jsonl"
        if not ev.exists():
            print(f"[跳过] {kb['id']}：没有评测集")
            continue

        print("=" * 82)
        print(f"[{kb['id']}] {kb['name']}  跑 {args.runs} 次")
        runs = []
        for r in range(1, args.runs + 1):
            res = one_run(idx, ev, gen=not args.no_gen)
            runs.append(res)
            if "total" in res:
                s = res["slowest"]
                print(f"  第{r}次: 中位 {res['total']['median']:>6}s | "
                      f"P95 {res['total']['p95']:>6}s | 最大 {res['total']['max']:>6}s | "
                      f"TTFT中位 {res['ttft']['median']:>5}s | 最慢: {s[0]}s {s[1]}")
            else:
                print(f"  第{r}次: 检索 中位 {res['retrieval']['median']}s | "
                      f"P95 {res['retrieval']['p95']}s | 最大 {res['retrieval']['max']}s")
        report[kb["id"]] = runs

    # 稳定性分析
    print("\n" + "=" * 82)
    print("稳定性分析：哪个指标在多次运行之间可信？")
    key = "retrieval" if args.no_gen else "total"
    print(f"\n  {'库':<12}{'指标':<10}{'各次值':<34}{'极差':>8}{'相对抖动':>10}")
    for kb_id, runs in report.items():
        for metric in ("median", "p95", "max"):
            vals = [r[key][metric] for r in runs if key in r and metric in r[key]]
            if not vals:
                continue
            spread = max(vals) - min(vals)
            rel = spread / statistics.median(vals) if statistics.median(vals) else 0
            flag = "  ← 稳定" if rel < 0.15 else ("  ← 抖动明显" if rel < 0.6 else "  ← 不可用")
            print(f"  {kb_id:<12}{metric:<10}{str(vals):<34}{spread:>8.2f}{rel:>10.1%}{flag}")

    if not args.no_gen:
        print("\n  最慢样本在多次运行中的分布（验证「长尾落在随机题上」）：")
        for kb_id, runs in report.items():
            qs = [r["slowest"][1] for r in runs]
            print(f"    {kb_id:<12} {qs}")
            print(f"                 {'（每次都同一题 → 可能是该题特性）' if len(set(qs)) == 1 else '（每次不同 → 确认为随机 API 长尾）'}")

    # ------------------------------------------------------------------
    # 合并所有运行的样本再统计 —— 这才是可信的做法
    #
    # 单次运行的 P95 在 n<50 时极不稳定（实测相对抖动 36%–68%）；
    # 连「中位数」在多次运行之间也会抖（实测最大 106%，因为整轮 API 变慢），
    # 所以**任何一个单次运行的延迟数字都不该拿来横向比较**。
    # 合并 R 次运行的 N×R 个样本后，统计量才稳定。
    # ------------------------------------------------------------------
    sample_key = "samples_retrieval" if args.no_gen else "samples_total"
    print("\n" + "=" * 82)
    print(f"合并统计（每个库把所有运行的单次延迟合并后计算，样本数 = 条数 × 运行次数）")
    print(f"  {'库':<12}{'样本数':>7}{'中位':>9}{'P90':>9}{'P95':>9}{'P99':>9}{'最大':>9}")
    pooled_summary = {}
    for kb_id, runs in report.items():
        xs = sorted(x for r in runs for x in r.get(sample_key, []))
        if not xs:
            continue
        n = len(xs)

        def q(p):
            return round(xs[min(n - 1, max(0, int(round(p * n)) - 1))], 2)

        row = {"n": n, "median": round(statistics.median(xs), 2),
               "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": round(max(xs), 2)}
        pooled_summary[kb_id] = row
        print(f"  {kb_id:<12}{n:>7}{row['median']:>9}{row['p90']:>9}"
              f"{row['p95']:>9}{row['p99']:>9}{row['max']:>9}")

    if pooled_summary:
        # 用合并后的中位数做横向比较（这是唯一站得住的比较）
        meds = {k: v["median"] for k, v in pooled_summary.items()}
        lo = min(meds.values())
        hi = max(meds.values())
        print(f"\n  合并中位延迟跨度：{lo}s – {hi}s（{hi/lo:.2f}×）")
        if hi / lo < 1.5:
            print(f"  → 三库延迟**没有实质差异**，之前的「dram 更慢」是单次 P95 的假象。")
        else:
            print(f"  → 存在实质差异，需要结合语料规模解释。")

    out_f = HERE / "kb" / "_latency_repeat.json"
    out_f.write_text(json.dumps({
        "runs_per_kb": args.runs, "no_gen": args.no_gen,
        "detail": report, "pooled": pooled_summary},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  明细已写入 kb/_latency_repeat.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
