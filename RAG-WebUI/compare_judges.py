# -*- coding: utf-8 -*-
"""对照实验：断言级「忠实度」vs 整体「答案正确性」，哪个真能测出答错。

上一轮判据验证的结论是忠实度不行：
  - 答案忠于「检索错的上下文」时，忠实度给满分（看不见答错）；
  - 措辞与原文不同的正确答案被判 0 分（冤枉答对）。
根因是忠实度测的是「答案是否忠于给定上下文」，不是「答案是否正确」。

这里加一个「答案正确性」裁判（拿答案与参考答案要点/资料比语义），
在同一批已知对错的样本上对比两者的区分度。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "RAG知识库Agent" / "demo"))
import faithfulness as F  # noqa: E402
import kb_agent as K  # noqa: E402

ACCURACY_PROMPT = """你是严格的技术答案评分员。判断【考生答案】是否正确回答了【问题】。

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
2. **如果答案讲的内容在【参考答案要点】里完全没有、也不等价，必须判 INCORRECT，哪怕它看起来很专业。**
3. **不要因为措辞与参考要点不同就判错**——只要意思一致就是 CORRECT。
4. **关键词命中检查**：先列出【参考答案要点】里的关键概念，
   再逐一到【考生答案】里找。**只要有一个关键概念在答案里完全找不到对应表述，
   等级不得高于 PARTIAL；若有多个关键概念缺失，或答案讲的是另一套内容，判 INCORRECT。**
   注意：这一条要优先于「答案是否流畅专业」——答得漂亮但没有参考要点里的概念，就是 INCORRECT。
5. 只输出 JSON，不要解释、不要 Markdown 代码块。

输出 JSON：
{"grade": "CORRECT|PARTIAL|INCORRECT|REFUSED", "reason": "一句话",
 "missing": "漏掉的关键概念，逗号分隔；没有则空字符串"}
"""

GRADE_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0, "REFUSED": 0.0}


def judge_accuracy(q: str, answer: str, ref: str, ctx: str) -> dict:
    obj = F._chat_json([
        {"role": "system", "content": ACCURACY_PROMPT},
        {"role": "user", "content":
            f"【问题】\n{q}\n\n【参考答案要点】\n{ref}\n\n{ctx}\n\n【考生答案】\n{answer}"},
    ])
    g = str(obj.get("grade", "")).upper().strip()
    if g not in GRADE_SCORE:
        g = "INCORRECT"
    return {"grade": g, "score": GRADE_SCORE[g],
            "reason": str(obj.get("reason") or "")[:200],
            "missing": str(obj.get("missing") or "")[:200]}


def main() -> int:
    rows_by_kb = {}
    for kb_id in ("ai_prep", "agent_eval"):
        d = json.loads((HERE / "kb" / kb_id / "eval" / "report_generation.json")
                       .read_text("utf-8"))
        chunks = {c["cid"]: c for c in json.loads(
            (HERE / "kb" / kb_id / "index" / "chunks.json").read_text("utf-8"))}
        ev = {c["q"]: c for c in (
            json.loads(l) for l in (HERE / "kb" / kb_id / "eval" / "eval_set.jsonl")
            .read_text("utf-8").splitlines() if l.strip() and not l.startswith("#"))}
        rows_by_kb[kb_id] = (d["cases"], chunks, ev)

    print("=" * 84)
    print("对照：同一批 10 条已知对错样本上，两种判据的表现\n")
    print(f"  {'人工':<6}{'忠实度':>8}{'正确性':>8}{'等级':<11}问题")
    good_f, bad_f, good_a, bad_a = [], [], [], []
    for kb_id, q, human, note in F.KNOWN:
        rows, chunks, ev = rows_by_kb[kb_id]
        row = next((x for x in rows if x["q"] == q), None)
        if row is None:
            continue
        hits = [{"cid": chunks[c]["cid"], "doc": chunks[c]["doc"],
                 "text": chunks[c]["text"]} for c in row["hits"] if c in chunks]
        fa = F.check_answer(q, row["answer"] or "", hits, {})
        ctx = F.build_context(hits)
        ac = judge_accuracy(q, row["answer"] or "", ev[q].get("ref", ""), ctx)

        print(f"  {human:<6}{fa['faithfulness']:>8}{ac['score']:>8}  "
              f"{ac['grade']:<11}{q[:34]}")
        if ac["grade"] != "CORRECT":
            print(f"        理由: {ac['reason'][:78]}")
        for lst, v in ((good_f, fa["faithfulness"]) if human == "答对"
                       else (bad_f, fa["faithfulness"]),):
            lst.append(v)
        for lst, v in ((good_a, ac["score"]) if human == "答对" else (bad_a, ac["score"]),):
            lst.append(v)

    print("\n" + "=" * 84)
    print("区分度对比（答对组均值 − 答错组均值，越大越好）")
    print(f"  忠实度:  答对 {statistics.mean(good_f):.3f} | 答错 {statistics.mean(bad_f):.3f} "
          f"| 差距 {statistics.mean(good_f)-statistics.mean(bad_f):+.3f}")
    print(f"  正确性:  答对 {statistics.mean(good_a):.3f} | 答错 {statistics.mean(bad_a):.3f} "
          f"| 差距 {statistics.mean(good_a)-statistics.mean(bad_a):+.3f}")
    print(f"\n  理想判据：答对组全 1.0（不冤枉），答错组全 0.0（不漏放）")
    print(f"  正确性判据误伤答对: {sum(1 for v in good_a if v < 1.0)}/{len(good_a)}")
    print(f"  正确性判据漏放答错: {sum(1 for v in bad_a if v > 0.5)}/{len(bad_a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
