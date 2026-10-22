# -*- coding: utf-8 -*-
"""交互式人工标注工具（多人标注 / IAA 用）。

设计要点
--------
1. **不显示任何 LLM 判据结论**。样本文件里本来就没有这些字段，
   本脚本也不会去读 report_accuracy。标注者只看到：题号、库、问题、参考答案要点、考生答案。
2. **自动保存 + 断点续标**。每标一条就落盘，Ctrl+C 也不会丢。
3. **可回溯**。输入 b 退回上一条重标。
4. 每位标注者一个文件（`_annotations_<名字>.json`），互不覆盖。
5. 支持只标一部分（`--limit 30`），方便分几次做完。

用法：
  python annotate.py --annotator alice              # 从第 1 条开始
  python annotate.py --annotator bob --limit 40     # 只标 40 条
  python annotate.py --annotator alice --review     # 复核已标过的
  python annotate.py --annotator alice --sheet      # 导出 Markdown 表格（可离线填）

等级：CORRECT / PARTIAL / INCORRECT。输入 1/2/3 或 c/p/i 均可。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
SAMPLE = KB / "_annotation_sample.json"

RUBRIC = """评分标准（先看这里，之后每题都按同一套标）：
  [1] CORRECT   参考要点里的关键概念在答案里都能找到对应表述，且无事实矛盾。
                措辞不同、更简略、少讲一个次要细节，都算 CORRECT。
  [2] PARTIAL   覆盖了部分关键概念，或漏了要点，但**没有说错**。
                含「答了同一主题但没答到点上」。
  [3] INCORRECT 有关键事实错误（与参考要点矛盾），
                或**答非所问**——通篇讲的是另一件事。
"""

KEYS = {
    "1": "CORRECT", "c": "CORRECT", "correct": "CORRECT",
    "2": "PARTIAL", "p": "PARTIAL", "partial": "PARTIAL",
    "3": "INCORRECT", "i": "INCORRECT", "incorrect": "INCORRECT",
}


def load_sample() -> list[dict]:
    if not SAMPLE.exists():
        raise SystemExit(f"找不到样本文件：{SAMPLE}\n请先跑 build_annotation_sample.py")
    return json.loads(SAMPLE.read_text("utf-8"))["cases"]


def load_done(path: Path) -> dict[str, str]:
    if path.exists():
        d = json.loads(path.read_text("utf-8"))
        return d.get("labels", {})
    return {}


def save(path: Path, labels: dict[str, str], annotator: str) -> None:
    path.write_text(json.dumps({
        "annotator": annotator,
        "desc": "人工标注结果。等级定义见 _annotation_sample.json 的 meta.rubric。",
        "n": len(labels),
        "labels": labels,
    }, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def show(case: dict, idx: int, total: int, done: int) -> None:
    print("\n" + "─" * 78)
    print(f"[{case['id']}]  第 {idx}/{total} 条（已标 {done}）  "
          f"来源 {case['source']}  类别 {case.get('cat') or '-'}")
    print(f"\n问题：{case['q']}")

    ref = case.get("ref") or ""
    if ref:
        print(f"\n参考答案要点：\n  {ref}")
    else:
        print("\n参考答案要点：\n  （本题为构造样本，无参考要点。"
              "按「答案是否正确回答了这个问题」判断即可。）")

    print(f"\n考生答案：\n  {case['answer']}")
    # 构造样本附上来源说明，帮助判断是不是"答非所问"
    if case.get("construction"):
        print(f"\n（本题构造方式：{case['construction']}）")
    print()


def do_sheet(sample: list[dict], out: Path) -> None:
    lines = ["# 人工标注表", "",
             "等级填 CORRECT / PARTIAL / INCORRECT。评分标准：", "",
             "| 等级 | 标准 |", "| --- | --- |",
             "| CORRECT | 参考要点的关键概念答案里都有，且无事实矛盾。措辞不同、更简略都算。 |",
             "| PARTIAL | 覆盖部分关键概念或漏要点，但没说错。 |",
             "| INCORRECT | 有关键事实错误，或答非所问讲了另一件事。 |", "",
             "| 题号 | 来源 | 问题 | 参考答案要点 | 考生答案 | 你的等级 |", "| --- | --- | --- | --- | --- | --- |"]
    for c in sample:
        def esc(s: str) -> str:
            return (s or "").replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(f"| {c['id']} | {c['source']} | {esc(c['q'])} | "
                     f"{esc(c.get('ref') or '（无）')} | {esc(c['answer'])} |  |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已导出标注表：{out.relative_to(HERE)}（{len(sample)} 条，最后一列留给你的等级）")


def main() -> int:
    ap = argparse.ArgumentParser(description="人工标注工具")
    ap.add_argument("--annotator", required=True, help="标注者名字（用于文件名，别写空格）")
    ap.add_argument("--limit", type=int, default=0, help="本次最多标几条（0=全部）")
    ap.add_argument("--review", action="store_true", help="复核已标过的")
    ap.add_argument("--sheet", action="store_true", help="导出 Markdown 表格而非交互标注")
    args = ap.parse_args()

    sample = load_sample()
    out = KB / f"_annotations_{args.annotator}.json"

    if args.sheet:
        do_sheet(sample, KB / f"_annotation_sheet_{args.annotator}.md")
        return 0

    labels = load_done(out)
    todo = [c for c in sample if args.review or c["id"] not in labels]

    print("=" * 78)
    print(f"标注者：{args.annotator}")
    print(f"合计 {len(sample)} 条，已标 {len(labels)} 条，本次待标 {len(todo)} 条")
    print(f"结果写入：{out.relative_to(HERE)}")
    print()
    print(RUBRIC)
    print("输入 1/2/3 或 c/p/i 评定等级；b=退回上一条；q=保存并退出。")
    print("=" * 78)

    if not todo:
        print("\n全部标完了。用 --review 可以复核。")
        return 0

    if args.limit:
        todo = todo[:args.limit]
        print(f"（--limit 生效，本次只标前 {len(todo)} 条）")

    i = 0
    while i < len(todo):
        case = todo[i]
        idx = [c["id"] for c in sample].index(case["id"]) + 1
        show(case, idx, len(sample), len(labels))
        try:
            raw = input("你的等级 (1/CORRECT, 2/PARTIAL, 3/INCORRECT, b, q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已中断，进度已保存。")
            break

        if raw == "q":
            break
        if raw == "b":
            if i > 0:
                i -= 1
                prev = todo[i]
                labels.pop(prev["id"], None)
                save(out, labels, args.annotator)
                print(f"已退回 {prev['id']}，请重标。")
            else:
                print("已经是第一条了。")
            continue

        g = KEYS.get(raw)
        if not g:
            print("⚠ 无法识别，请输入 1/2/3（或 c/p/i）。")
            continue

        labels[case["id"]] = g
        save(out, labels, args.annotator)     # 每标一条立刻落盘
        i += 1

    print(f"\n本次完成。累计已标 {len(labels)}/{len(sample)} 条。")
    if len(labels) < len(sample):
        print(f"继续标注：python annotate.py --annotator {args.annotator}")
    else:
        print("全部标完，可以跑 IAA 分析了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
