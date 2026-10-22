# -*- coding: utf-8 -*-
"""构建判据的「困难测试集」：提高 INCORRECT 基率，检验判据能否识别答非所问。

为什么要这样做
--------------
校准发现判据在 49 条真实作答上从不输出 INCORRECT，而人工标了 3 条（6%）。
三次提示词改动都没修好。判断根因是**低基率类别**：真实作答里「答非所问」太稀有，
判据缺少可参照的正例。

但直接往评测集里补「答非所问」的理论题，测的是**系统**而不是**判据**，
而且改评测集要重跑全量生成。所以这里另建一套专测判据的样本：

  构造「答非所问」的办法：把**真实的模型答案**（B 题的答案）错配到 A 题上。
  这样得到的是「语言流畅、内容专业、但完全答的不是这个问题」的答案——
  正是判据最难识别的形态，且比人工编造更像真实失败。

  同时加入「措辞不同但正确」的白话答案作对照，
  防止判据为了抓 INCORRECT 而过度严格（v2 版就犯过这个错）。

这批样本**不参与系统指标**，只回答一个问题：「判据能不能识别答非所问？」

用法：
  python build_hard_suite.py            # 生成 kb/_hard_suite.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"

# (kb, 被问的问题, 取哪个问题的答案, 人工等级, 理由)
# 分两类构造，因为它们的判定难度完全不同：
#   远域错配：A 与 B 讲的根本不是一件事 -> 明确 INCORRECT，判据必须识别
#   邻域错配：A 与 B 是同一小节里相邻的两个问题 -> 边界模糊，不计入召回率
SWAPS_FAR: list[tuple[str, str, str, str]] = [
    ("ai_prep", "混合检索里 RRF 的公式是什么？k 取多少？",
     "常用的向量相似度度量有哪几种？该选哪个？",
     "答案讲相似度度量，与 RRF 公式、k 值毫无关系。"),
    ("ai_prep", "KV Cache 的显存怎么估算？写出公式。",
     "自注意力的复杂度为什么是 O(n²)？",
     "答案讲注意力复杂度，与 KV Cache 显存公式无关。"),
    ("ai_prep", "幻觉有哪些缓解手段？哪种最有效？",
     "长上下文窗口越来越大，是不是就不需要 RAG 了？",
     "答案讲长上下文与 RAG 的关系，完全没提幻觉缓解。"),
    ("ai_prep", "生产环境里怎么防止 Agent 陷入无限循环？",
     "为什么现在的主流大模型都用 Pre-Norm 加 RMSNorm？",
     "答案讲模型结构，与 Agent 循环控制毫无关系。"),
    ("ai_prep", "长任务里 Agent 越跑越偏、忘记原来的目标，怎么防？",
     "向量维度越来越高会带来什么问题？",
     "答案讲维度灾难，与上下文漂移完全无关。"),
    ("agent_eval", "三层质量门分别测什么、什么频率？",
     "pass@k 和 pass^k 分别表示什么？",
     "答案讲 pass@k/pass^k 定义，未提及三层质量门。"),
    ("agent_eval", "最小的可用 Benchmark 要多少个 task？怎么分配？",
     "Agent 评测相比 LLM 评测，难点到底难在哪？",
     "答案讲评测难点，未回答 150 task 的数量与分配。"),
    ("agent_eval", "Agent 在对话里说「订单已创建」，但数据库里其实什么都没有。这类失败该怎么防？",
     "同一个 Agent、同一个任务跑十次结果都不一样，这该怎么处理？",
     "答案讲非确定性与多 trial，未涉及 Transcript/Outcome 验证。"),
    ("agent_eval", "LoCoMo 的评测维度有哪些？",
     "只靠分数为什么不够，还要人去看执行过程？",
     "答案讲读 transcript 的必要性，未列出 LoCoMo 的四个评测维度。"),
    ("agent_eval", "τ²-Bench 和 Tau-bench 分别测什么？",
     "给推荐/排序类 Agent 打分时，用什么指标衡量排序质量？",
     "答案讲排序指标，与两个 benchmark 各自测什么无关。"),
]

# 邻域错配：同属一节、话题相邻，边界模糊。人工倾向于判 INCORRECT（答的不是这个问题），
# 但也承认判 PARTIAL 有道理，所以**单独统计、不计入严格召回率**。
SWAPS_NEAR: list[tuple[str, str, str]] = [
    ("ai_prep", "生产环境里怎么防止 Agent 陷入无限循环？",
     "长任务里 Agent 越跑越偏、忘记原来的目标，怎么防？"),
    ("agent_eval", "最小的可用 Benchmark 要多少个 task？怎么分配？",
     "Capability Eval 和 Regression Eval 有什么区别？"),
    ("agent_eval", "τ²-Bench 和 Tau-bench 分别测什么？",
     "主流 Agent 评测 benchmark 各自覆盖什么能力？"),
    ("agent_eval", "做自动评分时，什么时候该用确定性判据，什么时候该让模型来打分？",
     "评测 Agent 时，Grader 在成本、速度和确定性上怎么权衡？"),
]

# 人工改写的白话答案：语义等价、措辞差异大，应判 CORRECT
# 数量与 SWAPS 大致相等，避免判据被推向「总判 INCORRECT」
PARAPHRASE: list[tuple[str, str, str, str]] = [
    ("ai_prep", "混合检索里 RRF 的公式是什么？k 取多少？",
     "就是对每路检索结果取「1 除以（常数加上它在这一路的排名）」再相加，常数一般用 60。",
     "用白话表达公式，不出现 Σ 符号。"),
    ("ai_prep", "RMSNorm 和 LayerNorm 有什么区别？",
     "LayerNorm 要减均值再除标准差，RMSNorm 只除以均方根、不减均值，所以少一步计算、更快，LLaMA 系列都用它。",
     "只说机制与收益，不讲完整公式。"),
    ("ai_prep", "近似最近邻（ANN）和维度灾难分别指什么？",
     "ANN 就是不再精确找最近的那个，稍微牺牲点准头换来快几个数量级；维度灾难是说维度越高，点和点之间距离都差不多，近邻和远邻分不出来。",
     "用口语解释两个概念，不用术语定义。"),
    ("ai_prep", "为什么让模型先想再答（CoT）和让它边想边调工具，不是一回事？",
     "前者只是自己闷头推理、不跟外界打交道；后者想完还要真的去调工具、看返回结果再决定下一步，多了「跟环境交互」这一环。",
     "强调有无外部交互，不出现 ReAct/Observation 等术语。"),
    ("agent_eval", "pass@k 和 pass^k 分别表示什么？",
     "前者是试 k 次里至少成功一次的概率，适合写代码这种一次过就行的任务；后者是 k 次全都得成功的概率，适合对客场景。",
     "用白话表述，不出现 pass@k 记号。"),
    ("agent_eval", "Transcript 和 Outcome 这两个概念的区别是什么？",
     "一个是模型在对话里说的内容（比如声称订单已创建），另一个是环境里真实发生的变化（比如数据库里到底有没有那行记录）。两者可能不一致。",
     "用「嘴上说的 vs 真实发生的」白话对照。"),
    ("agent_eval", "LLM Judge 的 Uncertain Escape Hatch 是什么？为什么必须有？",
     "就是给裁判模型留个「我也拿不准」的选项，不然它被逼着必须给分，就会在没把握的时候硬编一个分数出来。",
     "用口语解释「逃生门」及其必要性。"),
    ("agent_eval", "你怎么判断一次模型升级到底该不该上线？",
     "先在离线任务集上新旧各跑几遍做对比，看有没有显著退步；再让新版本接真实流量但不给用户用，观察一段时间；然后小流量灰度，只用护栏指标决定要不要停；最后按 1%→10%→50%→100% 逐步放开。",
     "用流程口述，不出现 Ship Gate/Offline Diff 等英文名。"),
    ("agent_eval", "同一个 Agent、同一个任务跑十次结果都不一样，这该怎么处理？",
     "这就是 Agent 的随机性，测一次没有意义。同一个任务多跑几遍（比如十次），报平均通过率和置信区间，而不是报单次结果。",
     "用白话说明多次试验报均值与区间。"),
    ("agent_eval", "给推荐/排序类 Agent 打分时，用什么指标衡量排序质量？",
     "看排在前面的对不对、排得靠不靠前要用 NDCG，还要看命中率、召回覆盖，另外得看推荐的东西够不够多样化、有没有覆盖到长尾。",
     "用白话点出 NDCG/命中/覆盖/多样性几个方向。"),
]


def load_answers() -> dict[str, dict[str, str]]:
    pool: dict[str, dict[str, str]] = {}
    for kb_id in ("ai_prep", "agent_eval"):
        gen = json.loads((KB / kb_id / "eval" / "report_generation.json")
                         .read_text("utf-8"))["cases"]
        pool[kb_id] = {x["q"]: (x["answer"] or "") for x in gen}
    return pool


def main() -> int:
    pool = load_answers()
    cases: list[dict] = []
    missing: list[str] = []

    for group, swaps in (("far", SWAPS_FAR), ("near", SWAPS_NEAR)):
        for kb_id, q, src_q, *rest in swaps:
            reason = rest[0] if rest else "同小节的相邻问题错配（边界情形）"
            if src_q not in pool[kb_id]:
                missing.append(f"{kb_id}：答案池里没有「{src_q}」")
                continue
            if q not in pool[kb_id]:
                missing.append(f"{kb_id}：题目池里没有「{q}」")
                continue
            cases.append({
                "kb": kb_id, "q": q, "answer": pool[kb_id][src_q],
                "human": "INCORRECT", "group": group,
                "human_reason": reason,
                "construction": f"把「{src_q}」的答案错配到本题（该答案对原题本身正确）",
            })

    for kb_id, q, ans, how in PARAPHRASE:
        if q not in pool[kb_id]:
            missing.append(f"{kb_id}：题目池里没有「{q}」")
            continue
        cases.append({
            "kb": kb_id, "q": q, "answer": ans,
            "human": "CORRECT", "group": "paraphrase",
            "human_reason": f"语义等价但措辞不同（{how}），应判 CORRECT，不应因措辞不同判错。",
            "construction": "人工改写为白话答案",
        })

    out = KB / "_hard_suite.json"
    out.write_text(json.dumps({
        "desc": "专测判据的困难样本集",
        "note": ("group=far：远域错配，答案讲的根本是另一件事，判据必须识别；"
                 "group=near：同小节相邻问题错配，边界模糊，单独统计不计入严格召回；"
                 "group=paraphrase：白话正确答案，用来检查判据不会过度严格。"
                 "本集不参与系统指标。"),
        "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"已写入 {out.relative_to(HERE)}，共 {len(cases)} 条")
    by_g = {}
    for c in cases:
        by_g.setdefault(c["group"], Counter())[c["human"]] += 1
    for g in ("far", "near", "paraphrase"):
        if g in by_g:
            print(f"  {g:<12}", dict(by_g[g]))
    if missing:
        print("\n⚠ 以下条目未生成：")
        for m in missing:
            print("   ", m)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
