# -*- coding: utf-8 -*-
"""为新增知识库构建评测集。

沿用 demo 的 build_eval_set.py 规范（字段与四类划分一致），并保留它的两道校验：
  1. gold_cids 必须真实存在于索引中；
  2. 精确匹配类（A）的 gold chunk 里必须能找到问题中的关键实体
     —— 防的是 README 里记载的那种标注错误：标错了 chunk，题目恒不命中，
        指标静默偏低而不报错。

用法：
  python build_eval.py                # 构建全部
  python build_eval.py --only ai_prep
  python build_eval.py --check-only   # 只校验不写文件
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_ROOT = HERE / "kb"

# ---------------------------------------------------------------------------
# 评测集定义：(问题, [gold_cids], 是否可回答, 参考答案要点, 类别)
# 类别：A-精确匹配 / B-语义泛化 / C-聚合 / D-不可回答
# ---------------------------------------------------------------------------
EVAL_SETS: dict[str, list[tuple]] = {}

# ============================ Agent 评测手册 ============================
EVAL_SETS["agent_eval"] = [
    # ---- A. 精确匹配：术语/缩写，关键词检索主场 ----
    ("Transcript 和 Outcome 这两个概念的区别是什么？",
     ["Agent评测手册#3", "Agent评测手册#4"], True,
     "Transcript 是 Trial 的完整对话记录（messages + tool calls）；Outcome 是执行后的环境状态和关键指标。Agent 说的≠实际做的，必须验证 Outcome",
     "A-精确匹配"),
    ("pass@k 和 pass^k 分别表示什么？",
     ["Agent评测手册#7"], True,
     "pass@k 是 k 次尝试中至少 1 次成功的概率；pass^k 是 k 次全部成功的概率，适合面向用户的 Agent",
     "A-精确匹配"),
    ("Capability Eval 和 Regression Eval 有什么区别？",
     ["Agent评测手册#7"], True,
     "Capability 问「这 Agent 能做什么」，期望 pass rate 30-50%；Regression 问「还保持着原来的能力吗」，期望接近 100%",
     "A-精确匹配"),
    ("Eval Harness 和 Agent Harness 有什么区别？",
     ["Agent评测手册#3", "Agent评测手册#4"], True,
     "Eval Harness 是跑评测的基础设施（环境重置、Trial 执行、结果收集）；Agent Harness 是把模型变成 Agent 的框架，两者要分开测",
     "A-精确匹配"),
    ("LLM Judge 的 Uncertain Escape Hatch 是什么？为什么必须有？",
     ["Agent评测手册#6"], True,
     "给 judge 一个「不确定」的出口（如 INSUFFICIENT_INFO），强制它不确定就说不确定；否则等于强迫 LLM 在不确定时乱打分",
     "A-精确匹配"),
    ("三层质量门分别测什么、什么频率？",
     ["Agent评测手册#13", "Agent评测手册#14"], True,
     "Layer 1 安全底线（blacklist/PII，每条 transcript 实时）、Layer 2 业务 KPI（每天定时）、Layer 3 离线质量（约 150 task suite，每周 + 每次模型升级）",
     "A-精确匹配"),
    ("最小的可用 Benchmark 要多少个 task？怎么分配？",
     ["Agent评测手册#13"], True,
     "150 个：Capability 50 + Regression 50 + Adversarial 30 + Multi-turn Hard 20",
     "A-精确匹配"),
    ("τ²-Bench 和 Tau-bench 分别测什么？",
     ["Agent评测手册#17", "Agent评测手册#18"], True,
     "τ²-Bench 测航空/零售等真实业务场景（工具+状态+政策）；Tau-bench 测客服场景多轮交互",
     "A-精确匹配"),
    ("LoCoMo 的评测维度有哪些？",
     ["Agent评测手册#18"], True,
     "长时记忆一致性、事实更新后回答是否一致、多轮任务完成情况、时序推理",
     "A-精确匹配"),

    # ---- B. 语义泛化：问法不含原文关键词 ----
    ("Agent 在对话里说「订单已创建」，但数据库里其实什么都没有。这类失败该怎么防？",
     ["Agent评测手册#4", "Agent评测手册#20"], True,
     "Transcript 欺骗。至少 50% 的 Check 必须验证 Outcome（DB state、API 记录、文件变更），不能只看 Transcript",
     "B-语义泛化"),
    ("做自动评分时，什么时候该用确定性判据，什么时候该让模型来打分？",
     ["Agent评测手册#5", "Agent评测手册#6"], True,
     "确定性优先：code-based 做守门员拦明显错误，LLM judge 只评语气/风格/完整性等主观维度，human 做校准和关键决策；三种是层层递进不是替换",
     "B-语义泛化"),
    ("你怎么判断一次模型升级到底该不该上线？",
     ["Agent评测手册#15", "Agent评测手册#24"], True,
     "四级 Ship Gate：Offline Diff（paired comparison + bootstrap CI）→ Shadow Traffic（24h，read divergent transcripts）→ Guardrail-gated A/B → Phased Rollout",
     "B-语义泛化"),
    ("为了让 Agent 看似成功，它学会说漂亮话而不是真解决问题，这种情况怎么发现？",
     ["Agent评测手册#27"], True,
     "这就是 reward hacking（UC Berkeley 发现 8 大 benchmark 都能被 hack 到接近 100%）；对策是 Grader 必须验证 Outcome，且 eval 本身要持续演进（living artifact + 定期 read transcripts）",
     "B-语义泛化"),
    ("只靠分数为什么不够，还要人去看执行过程？",
     ["Agent评测手册#29", "Agent评测手册#30"], True,
     "分数只告诉你过了没有，不告诉你原因；失败可能集中在某类任务上，不读 transcripts 永远发现不了那个 pattern",
     "B-语义泛化"),
    ("同一个 Agent、同一个任务跑十次结果都不一样，这该怎么处理？",
     ["Agent评测手册#19"], True,
     "非确定性是 Agent 评测的根本挑战之一；应对是每个 task 跑多个 trial（至少 10 次），报均值和置信区间",
     "B-语义泛化"),
    ("多轮任务里前面一个小偏差，到后面变成大错，评测该怎么设计？",
     ["Agent评测手册#19"], True,
     "错误传播（蝴蝶效应）；难点不在第一轮对不对，而在错了之后能不能自己发现并修正——所以要评中间步骤的纠错能力，不只评最终结果",
     "B-语义泛化"),
    ("怎么用一个具体的数字说明「每次都要好」比「至少对一次」要求高得多？",
     ["Agent评测手册#7"], True,
     "pass@k vs pass^k：k=3 时单次成功率 75%，连续 3 次都满意的用户只有 42%",
     "B-语义泛化"),
    ("给推荐/排序类 Agent 打分时，用什么指标衡量排序质量？",
     ["Agent评测手册#21"], True,
     "NDCG@k（考虑位置和相关性等级）、MRR、Recall@k、HitRate@k，另外还有 Intra-list Diversity 与 Catalog Coverage 衡量多样性覆盖",
     "B-语义泛化"),
    ("构建评测集时，任务该从哪里来、怎么保证标注质量？",
     ["Agent评测手册#12"], True,
     "任务来源七步法：Mine（从 production logs／用户投诉／CS escalation 挖掘）/ Cluster（embedding 聚类）/ Sample（按 cluster 采样保证 coverage）/ Reference（标注什么是 pass + reference trace）/ SME + IAA（至少两位 SME 标注，IAA<0.6 就重写 task）/ Balance / Adversarial",
     "B-语义泛化"),

    # ---- C. 聚合：需要跨多条信息 ----
    ("评测 Agent 时，Grader 在成本、速度和确定性上怎么权衡？",
     ["Agent评测手册#5"], True,
     "Code-based 完全确定、毫秒级、极低成本；LLM Judge 非确定、秒级、中等成本；Human 非确定、小时/天级、极高成本",
     "C-聚合"),
    ("Agent 评测相比 LLM 评测，难点到底难在哪？",
     ["Agent评测手册#19", "Agent评测手册#20"], True,
     "五个挑战：非确定性、错误传播（蝴蝶效应）、环境漂移、Reward Hacking、Transcript 欺骗",
     "C-聚合"),
    ("从跑评测到真的上线，完整流程有哪些阶段？",
     ["Agent评测手册#9", "Agent评测手册#15"], True,
     "Anthropic 8 步路线图（Step 0 起步 20-50 task 到 Step 8 living artifact）；上线用四级 Ship Gate",
     "C-聚合"),
    ("评测里有哪些看起来没问题、其实很致命的坑？",
     ["Agent评测手册#11", "Agent评测手册#16"], True,
     "0% pass@100 往往是 task 写错了；不要只看 tool call sequence；LLM judge 不加 escape hatch；用 primary metric 做 early stop 是隐式 p-hacking",
     "C-聚合"),
    ("主流 Agent 评测 benchmark 各自覆盖什么能力？",
     ["Agent评测手册#17", "Agent评测手册#18"], True,
     "SWE-bench Verified（软件工程）、WebArena（真实网页多步）、GAIA（通用助手）、AgentBench（8 类环境广度）、τ²-Bench（真实业务办事）、Tau-bench（客服多轮）、LoCoMo（长对话记忆）",
     "C-聚合"),

    # ---- D. 不可回答：资料里确实没有（测幻觉） ----
    ("Claude 3.5 Sonnet 的 API 每百万 token 多少钱？", [], False,
     "手册没有 API 定价，应回答资料不足", "D-不可回答"),
    ("DeepSeek-V3 一共有多少参数？", [], False,
     "手册没有 DeepSeek-V3 参数量，应回答资料不足", "D-不可回答"),
    ("SWE-bench Verified 上目前最高分是多少？", [], False,
     "手册只说了该 benchmark 测什么，没有具体分数，应回答资料不足", "D-不可回答"),
    ("我们公司上个季度 Agent 业务的营收是多少？", [], False,
     "手册没有公司经营数据，应回答资料不足", "D-不可回答"),
    ("GPT-4o 的上下文窗口具体是多少 token？", [], False,
     "手册没有 GPT-4o 的具体窗口数值，应回答资料不足", "D-不可回答"),
    ("LangSmith 和 Langfuse 的订阅价格差多少？", [], False,
     "手册没有工具定价，应回答资料不足", "D-不可回答"),
]

# ============================ AI Agent 备考清单 ============================
EVAL_SETS["ai_prep"] = [
    # ---- A. 精确匹配 ----
    ("自注意力的复杂度为什么是 O(n²)？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#4", "AI_Agent_大模型应用_技术一面_备考知识清单#5"], True,
     "注意力要对每个 token 与所有 token 两两计算相似度，序列长度 n 对应 n×n 个组合；时间与空间复杂度都是 O(n²·d)",
     "A-精确匹配"),
    ("KV Cache 的显存怎么估算？写出公式。",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#8"], True,
     "2 × L × n_kv_heads × d_head × seq_len × batch × bytes_per_element；2 是 K/V 两份，bytes 是精度（FP16=2，INT8=1）",
     "A-精确匹配"),
    ("MQA 和 GQA 是为了省什么？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#7"], True,
     "省 KV Cache 显存，不是省参数量；LLaMA-2 70B 用 GQA 把 KV Cache 缩到 MHA 的约 1/8",
     "A-精确匹配"),
    ("RMSNorm 和 LayerNorm 有什么区别？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#14"], True,
     "RMSNorm 去掉均值中心化，只用均方根 x/sqrt(mean(x²)+ε)·γ；少一次减法、少一个 β 参数，更快且效果基本不降",
     "A-精确匹配"),
    ("混合检索里 RRF 的公式是什么？k 取多少？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#61", "AI_Agent_大模型应用_技术一面_备考知识清单#62"], True,
     "score(d) = Σ_i 1/(k + rank_i(d))，k 通常取 60；只用排名不用分数，规避两路分数不可比",
     "A-精确匹配"),
    ("近似最近邻（ANN）和维度灾难分别指什么？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#60"], True,
     "ANN 牺牲少量精度换数量级速度提升；维度灾难指维度升高后样本稀疏、距离差异缩小，最近邻与远邻难以区分",
     "A-精确匹配"),
    ("常用的向量相似度度量有哪几种？该选哪个？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#54", "AI_Agent_大模型应用_技术一面_备考知识清单#55"], True,
     "余弦相似度、点积、欧氏距离；关键是选择和 embedding 模型训练时一致的度量方式，否则召回明显掉",
     "A-精确匹配"),

    # ---- B. 语义泛化 ----
    ("一个模型只会按你的指令回一次话，另一个会自己决定下一步做什么，差别在哪？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#29"], True,
     "普通 LLM 应用是「一次调用得到答案」，Agent 是「模型自己决定下一步做什么，不需要人逐步指挥」",
     "B-语义泛化"),
    ("为什么让模型先想再答（CoT）和让它边想边调工具，不是一回事？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#32", "AI_Agent_大模型应用_技术一面_备考知识清单#33"], True,
     "CoT 只在推理空间里想，ReAct 的 Thought 之后必须接 Action（调工具）并读 Observation；CoT 可以嵌进 ReAct 的 Thought 里",
     "B-语义泛化"),
    ("如果资料里根本没有答案，怎么让模型老实说不知道、而不是硬编一个？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#21"], True,
     "Prompt 里明确拒答指令（资料中没有就直接回答「资料中未提及」）；配合 RAG 提供证据、结构化输出约束、要求引用，实际项目里组合使用",
     "B-语义泛化"),
    ("模型离线指标很好看，一上线就掉，通常是什么原因？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#21"], True,
     "要区分检索侧幻觉和生成侧幻觉：前者是没检索到或排错，后者是有资料但模型没按资料答；上线掉还要查数据泄漏与分布漂移",
     "B-语义泛化"),
    ("换了新的向量模型之后，为什么不能直接把旧的向量覆盖掉？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#56"], True,
     "新旧 query 与文档向量不在同一语义空间，混用会导致召回崩坏；应做 versioned index，新老索引并行、摄取双写、灰度对比 Recall@k 后再下线旧索引",
     "B-语义泛化"),
    ("为什么向量检索之外还要留一路关键词检索？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#60", "AI_Agent_大模型应用_技术一面_备考知识清单#61", "AI_Agent_大模型应用_技术一面_备考知识清单#62"], True,
     "向量擅长语义泛化但会糊掉专有名词/型号/错误码，BM25 擅长精确匹配；两路召回互补再融合",
     "B-语义泛化"),
    ("Recall@k 已经很高了，为什么用户还是觉得答不准？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#62", "AI_Agent_大模型应用_技术一面_备考知识清单#63", "AI_Agent_大模型应用_技术一面_备考知识清单#64"], True,
     "Recall 解决「召回全不全」，排序准不准要靠 rerank；融合后必须接 rerank",
     "B-语义泛化"),
    ("用户问法和文档里的说法完全不一样，怎么提高命中率？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#64", "AI_Agent_大模型应用_技术一面_备考知识清单#65"], True,
     "查询改写（Multi-Query）与 HyDE（先生成假想答案再检索）；改写后要做一致性检查，并保留原始 query 作为一路通道",
     "B-语义泛化"),
    ("怎么把「让模型知道新知识」和「让模型学会新行为」这两件事分开处理？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#23"], True,
     "知道新知识用 RAG（改文档即可，实时生效、可追溯）；学会新行为用微调（改变行为模式，需要 GPU + 高质量标注）；按格式输出先试 Prompt",
     "B-语义泛化"),
    ("长任务里 Agent 越跑越偏、忘记原来的目标，怎么防？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#50", "AI_Agent_大模型应用_技术一面_备考知识清单#51"], True,
     "上下文漂移；做法包括维护任务清单、阶段性复述目标、Observation 摘要、上下文压缩按优先级裁剪",
     "B-语义泛化"),

    # ---- C. 聚合 ----
    ("微调、RAG、提示工程三者该怎么选？各自的代价是什么？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#23"], True,
     "一句话判据：知道新知识用 RAG；学会新行为用微调；按格式输出先试 Prompt。RAG 不改模型、成本低可追溯；Prompt 零成本但换版本可能失效；微调需 GPU+标注但能稳定行为",
     "C-聚合"),
    ("幻觉有哪些缓解手段？哪种最有效？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#21"], True,
     "五板斧按优先级：RAG（最有效）、Function Calling、Structured Output + JSON Schema、Prompt 明确拒答、多轮验证/交叉校验；要区分检索侧与生成侧幻觉",
     "C-聚合"),
    ("为什么现在的主流大模型都用 Pre-Norm 加 RMSNorm？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#14"], True,
     "Pre-Norm 让残差路径无归一化、梯度直接流过，训练稳定且深层不需 warmup；RMSNorm 去掉均值中心化，更快且效果基本不降（LLaMA 系全用）",
     "C-聚合"),
    ("生产环境里怎么防止 Agent 陷入无限循环？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#44"], True,
     "多层保险：max_iterations 硬上限、重复检测（工具名+参数哈希）、进度判断、Observation 摘要、失败工具冷却、Critic/Controller 节点；「循环是控制问题，不只是提示词问题」",
     "C-聚合"),
    ("长上下文窗口越来越大，是不是就不需要 RAG 了？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#20", "AI_Agent_大模型应用_技术一面_备考知识清单#21"], True,
     "不是。长上下文的有效利用率远低于标称值，且成本和延迟是平方级的，所以长窗口与 RAG 互补而非替代",
     "C-聚合"),
    ("混合检索的两路分数怎么融合？为什么推荐 RRF？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#61", "AI_Agent_大模型应用_技术一面_备考知识清单#62"], True,
     "融合方法有 RRF、加权求和、Learning to Rank；RRF 用 score(d)=Σ1/(k+rank_i(d))，k=60，只用排名不用分数，规避两路分数不可比",
     "C-聚合"),
    ("手撕自注意力时，哪些细节最容易写错？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#118", "AI_Agent_大模型应用_技术一面_备考知识清单#119"], True,
     "缩放因子 1/sqrt(d_k)、causal mask 的位置、softmax 的维度 dim=-1",
     "C-聚合"),
    ("向量维度越来越高会带来什么问题？",
     ["AI_Agent_大模型应用_技术一面_备考知识清单#60"], True,
     "维度灾难：样本变稀疏、距离差异缩小，最近邻与远邻难区分，暴力搜索和索引都变难（这也是 ANN 存在的原因）",
     "C-聚合"),

    # ---- D. 不可回答 ----
    ("Qwen3-8B 一共有多少参数、上下文窗口多大？", [], False,
     "清单没有 Qwen3-8B 的具体参数量与窗口，应回答资料不足", "D-不可回答"),
    ("MTEB 中文榜单上 bge-m3 排第几？", [], False,
     "清单没有 MTEB 排名数据，应回答资料不足", "D-不可回答"),
    ("OpenAI 的 embedding 接口每百万 token 多少钱？", [], False,
     "清单没有 API 定价，应回答资料不足", "D-不可回答"),
    ("我们公司去年因为设备故障损失了多少产能？", [], False,
     "清单没有公司生产数据，应回答资料不足", "D-不可回答"),
    ("DeepSeek-V3 训练用了多少张 H800 卡？", [], False,
     "清单没有 DeepSeek-V3 的训练硬件信息，应回答资料不足", "D-不可回答"),
]


# ---------------------------------------------------------------------------
# 校验与写出
# ---------------------------------------------------------------------------
ENT = re.compile(r"[A-Za-z][A-Za-z0-9\-\./^]{1,}|[\u4e00-\u9fff]{2,}")
STOP = re.compile(r"^[的是什么怎么为吗呢了和与或及在把对被让使从到等这那很就都也还要会能可]+$")


def entities_of(q: str) -> list[str]:
    """从问题里抽可比对的实体。

    英文/型号整体比对；中文不能整串比对（问题里是「自注意力的复杂度为什么是」，
    而 chunk 里是分行的「自注意力」），所以取 2–4 字的 n-gram，
    只要有一个出现在 gold chunk 里就算通过。
    """
    ents: list[str] = []
    for tok in ENT.findall(q):
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\./^]*", tok):
            if len(tok) >= 2:
                ents.append(tok)
        else:
            if STOP.match(tok):
                continue
            for n in (4, 3, 2):
                for i in range(len(tok) - n + 1):
                    gram = tok[i:i + n]
                    if not STOP.match(gram):
                        ents.append(gram)
    return ents


def suspects(kb_id: str, cases: list[tuple]) -> list[str]:
    """可疑标注提示（不阻断写出）。

    针对一种很难自查的错误：答案被 chunk 边界切开，完整问答其实在相邻块里，
    但只标了其中一块。实测踩到过一次——「混合检索的两路分数怎么融合？
    为什么推荐 RRF？」标了 #62（只是 RRF 公式的重叠延续），而完整问答在 #61。
    这类错误能让指标反向（vector 排名>10 而 hybrid rank=4），很难从数字上察觉。

    判据：问题的**连续子串**（去标点后 ≥10 字）出现在相邻的非 gold chunk 里。
    这比「关键词出现在相邻块」精确得多——后者会命中「Agent」「RAG」这类
    到处都是的词，报警太多就等于没有报警。
    """
    chunks = json.loads((KB_ROOT / kb_id / "index" / "chunks.json").read_text("utf-8"))
    text_of = {c["cid"]: c["text"] for c in chunks}
    order = [c["cid"] for c in chunks]
    pos = {cid: i for i, cid in enumerate(order)}
    notes: list[str] = []
    MIN_SPAN = 12                      # 太短会命中「Transcript」这类普通词

    def norm(s: str) -> str:
        return re.sub(r"[\s，。！？、；：“”‘’（）()《》—…·\-]+", "", s)

    for q, gold, ans, ref, cat in cases:
        if not gold:
            continue
        qn = norm(q)
        if len(qn) < MIN_SPAN:
            continue
        gold_pos = [pos[g] for g in gold if g in pos]
        if not gold_pos:
            continue
        lo, hi = min(gold_pos) - 1, max(gold_pos) + 1
        near = [order[i] for i in range(max(0, lo), min(len(order), hi + 1))
                if order[i] not in gold]
        for cid in near:
            cn = norm(text_of[cid])
            for span in range(len(qn), MIN_SPAN - 1, -1):
                found = None
                for i in range(len(qn) - span + 1):
                    if qn[i:i + span] in cn:
                        found = qn[i:i + span]
                        break
                if found:
                    notes.append(f"{q[:36]}\n             gold={gold}\n"
                                 f"             相邻未标块 {cid} 含有问题原文片段"
                                 f"「{found}」")
                    break
            else:
                continue
            break
    return notes


def check(kb_id: str, cases: list[tuple]) -> list[str]:
    chunks = json.loads((KB_ROOT / kb_id / "index" / "chunks.json").read_text("utf-8"))
    text_of = {c["cid"]: c["text"] for c in chunks}
    problems: list[str] = []

    # 校验 1：gold_cid 必须存在于索引
    for q, gold, *_ in cases:
        for g in gold:
            if g not in text_of:
                problems.append(f"[gold 不存在] {g}  (问题: {q[:34]})")

    # 校验 2：A 类问题的关键实体必须出现在 gold chunk 里
    for q, gold, ans, ref, cat in cases:
        if not gold or not cat.startswith("A"):
            continue
        ents = entities_of(q)
        if not ents:
            continue
        ok = any(any(e.lower() in text_of[g].lower() for e in ents) for g in gold)
        if not ok:
            problems.append(f"[实体不匹配] 问题: {q}\n             gold={gold}  "
                            f"抽取实体={ents[:8]}")
    return problems


def write_set(kb_id: str, cases: list[tuple]) -> None:
    name = {"agent_eval": "Agent 评测手册",
            "ai_prep": "AI Agent 备考清单"}.get(kb_id, kb_id)
    lines = [
        f"# 评测集：{name}",
        "# 字段：q=问题, gold_cids=正确答案所在 chunk, answerable=资料是否足以回答,",
        "#       ref=参考答案要点, cat=类别",
        "# 类别：A-精确匹配（关键词检索主场）/ B-语义泛化（向量检索主场）",
        "#       C-聚合（跨多条）/ D-不可回答（测幻觉）",
        "# 由 RAG-WebUI/build_eval.py 生成，含 gold 存在性与实体一致性校验。",
    ]
    for q, gold, ans, ref, cat in cases:
        lines.append(json.dumps({"q": q, "gold_cids": gold, "answerable": ans,
                                 "ref": ref, "cat": cat}, ensure_ascii=False))
    out = KB_ROOT / kb_id / "eval" / "eval_set.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cnt = Counter(c for *_, c in cases)
    print(f"\n[{kb_id}] 已写入 {out.relative_to(HERE)}")
    print(f"  共 {len(cases)} 条：", "  ".join(f"{k} {v}" for k, v in sorted(cnt.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description="构建新知识库的评测集")
    ap.add_argument("--only", default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    failed = False
    for kb_id, cases in EVAL_SETS.items():
        if args.only and kb_id != args.only:
            continue
        print(f"=== {kb_id}：{len(cases)} 条 ===")
        problems = check(kb_id, cases)
        if problems:
            failed = True
            print("  ❌ 校验失败：")
            for p in problems:
                print(f"    {p}")
        else:
            print("  ✅ gold_cids 全部存在；A 类实体校验通过")
        for n in suspects(kb_id, cases):
            print(f"  ⚠ 建议复核（答案可能被切到相邻块）：\n    {n}")
        if not args.check_only and not problems:
            write_set(kb_id, cases)

    if failed:
        print("\n⚠ 有校验失败项，未写出对应文件。修正后重跑。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
