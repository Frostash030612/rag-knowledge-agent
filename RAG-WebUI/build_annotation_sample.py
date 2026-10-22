# -*- coding: utf-8 -*-
"""为「多人标注 / IAA」准备标注样本。

为什么需要它
------------
所有正确性指标都建立在一个 LLM 判据上，而判据在 49 条人工标注上 Kappa 只有 0.516。
要判断这个判据够不够用，需要**多个标注者对同一批样本独立标注**，算出标注者间一致性（IAA）。
如果两个人都难达成一致，说明"评分标准本身模糊"，那么任何判据的 Kappa 都有天花板。

关键设计：**标注文件里不含判据结论**
--------------------------------------
如果标注者能看到 `judge` 的等级，就会被锚定，IAA 随即失去意义。
所以本脚本输出样本时剥掉 judge 的 grade / missing / reason，
只留：题号、知识库、问题、参考答案要点、考生答案（、可选：当次检索到的资料）。

样本构成
--------
  - 三个库全部「可作答」的生成结果（dram 31 + ai_prep 25 + agent_eval 24 = 80 条）
  - 困难集里的 24 条（远域错配 10 / 邻域错配 4 / 白话正确 10），
    用来保证 INCORRECT 与"措辞不同但正确"这两类都有足够样本
总量约 104 条。顺序用固定种子打乱，避免"总是先看到某个库"的顺序偏差。

用法：
  python build_annotation_sample.py                 # 生成 sample
  python build_annotation_sample.py --with-hits     # 附上当次检索到的资料（标注更难但更接近判据视角）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
OUT = KB / "_annotation_sample.json"

# 标注者**不该**看到的字段（会锚定判断）
HIDE = {"grade", "score", "reason", "missing", "missing_list", "missing_ungrounded",
        "covered_list", "judged_answerable", "answer_coverage"}


def load_pool() -> list[dict]:
    """收集所有待标注样本，并剥掉判据结论。"""
    pool: list[dict] = []

    for kb_id in ("dram", "ai_prep", "agent_eval"):
        gen = json.loads((KB / kb_id / "eval" / "report_generation.json")
                         .read_text("utf-8"))["cases"]
        for x in gen:
            if not x.get("judged_answerable"):
                continue          # 拒答题没有答案可评，不纳入
            pool.append({
                "source": f"gen:{kb_id}",
                "kb": kb_id,
                "cat": x.get("cat", ""),
                "q": x["q"],
                "ref": x.get("ref", ""),
                "answer": x.get("answer") or "",
                "hits": x.get("hits") or [],
            })

    hard = json.loads((KB / "_hard_suite.json").read_text("utf-8"))["cases"]
    for c in hard:
        pool.append({
            "source": f"hard:{c.get('group', '')}",
            "kb": c["kb"],
            "cat": c.get("group", ""),
            "q": c["q"],
            "ref": "",            # 困难集没有 ref（它是按构造给出人工等级的）
            "answer": c["answer"],
            "hits": [],
            "construction": c.get("construction", ""),
        })

    # 困难集的人工等级已知，但那是"造样本时设定的标签"，等同于第一个标注者的答案，
    # 不能放进标注样本（否则第二个标注者会被锚定）。
    # 所以困难集的 ref/等级都不输出，只在 IAA 分析时由 build_hard_suite 的原始文件对照。
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 IAA 标注样本")
    ap.add_argument("--with-hits", action="store_true",
                    help="附上当次检索到的资料（更接近判据的视角，但标注负担更大）")
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    pool = load_pool()

    # 固定种子打乱：可复现，且避免顺序偏差
    random.Random(args.seed).shuffle(pool)

    for i, item in enumerate(pool, 1):
        item["id"] = f"A{i:03d}"
        # 再兜底剥一次，防止今后有人往 pool 里加了判据字段
        for k in list(item.keys()):
            if k in HIDE:
                item.pop(k)
        if not args.with_hits:
            item.pop("hits", None)

    meta = {
        "desc": "多人标注用样本。**不含任何 LLM 判据结论**，避免锚定。",
        "n": len(pool),
        "seed": args.seed,
        "with_hits": args.with_hits,
        "rubric": {
            "CORRECT": "参考要点里的关键概念在答案里都能找到对应表述，且无事实矛盾。措辞不同、更简略都算。",
            "PARTIAL": "覆盖了部分关键概念，或漏了要点，但没有说错。含「答了同一主题但没答到点上」。",
            "INCORRECT": "有关键事实错误（与参考要点矛盾），或答非所问地讲了完全另一套内容。",
        },
        "grades_order": ["INCORRECT", "PARTIAL", "CORRECT"],
        "说明": [
            "seed 固定，重跑会得到相同顺序，便于两位标注者对照。",
            "source 字段仅供事后分组分析（gen:<库> / hard:<组>），不暗示答案。",
            "每题只看到 ref 与答案；不要参考其他题的判定。",
            "独立标注，不要互相讨论具体题目（讨论会抬高 IAA，失去估计意义）。",
        ],
    }

    OUT.write_text(json.dumps({"meta": meta, "cases": pool},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 单独导出判据侧的辅助信息（只给 IAA 分析脚本用，标注者绝不能看到）----
    #
    # 两件事需要它：
    #   1) 困难集的 ref 是空的，判据要评价需要 ref；
    #   2) 困难集的人工等级（造样本时设定的）要先藏起来，分析时再加载。
    helpers: dict[str, dict] = {}
    # 生成结果的 ref / hits 从原始报告取
    for kb_id in ("dram", "ai_prep", "agent_eval"):
        gen = json.loads((KB / kb_id / "eval" / "report_generation.json")
                         .read_text("utf-8"))["cases"]
        for x in gen:
            if x.get("judged_answerable"):
                helpers[f"gen:{kb_id}\x00{x['q']}"] = {
                    "ref": x.get("ref", ""), "hits": x.get("hits") or []}
    # 困难集的 ref 取被错配问题的 ref（远/邻域用原题 ref），等级单独存
    hard = json.loads((KB / "_hard_suite.json").read_text("utf-8"))["cases"]
    hard_key = {}
    for c in hard:
        hard_key[f"{c['kb']}\x00{c['q']}\x00{c['answer'][:40]}"] = {
            "human_grade_by_construct": c["human"],
            "group": c.get("group", ""),
            "construction": c.get("construction", ""),
        }
    helper_f = KB / "_annotation_helpers.json"
    helper_f.write_text(json.dumps({
        "desc": "IAA 分析用的辅助信息：困难集 ref、按构造预设的人工等级、检索 hits。"
                "**标注者不得查看本文件**，否则破坏了独立标注。",
        "gen_helpers": helpers,
        "hard_keys": hard_key,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  已写入 {helper_f.relative_to(HERE)}（供分析用，标注者不可看）")

    print(f"已写入 {OUT.relative_to(HERE)}：{len(pool)} 条（seed={args.seed}）")
    print("  来源分布:", dict(Counter(c["source"] for c in pool)))
    print("  含 hits:", args.with_hits)
    print(f"  字段: {sorted(pool[0].keys())}")
    assert not (HIDE & set(pool[0].keys())), "样本里仍含判据字段！"
    print("  ✅ 已确认样本中不含判据结论字段")
    return 0


if __name__ == "__main__":
    sys.exit(main())
