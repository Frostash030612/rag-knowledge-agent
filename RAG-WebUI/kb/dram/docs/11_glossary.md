# 术语与缩略语对照表

本表用于快速查阅半导体制造与数据分析中的常见缩写。

## 1. 器件与工艺

| 缩写 | 全称 | 中文含义 |
| --- | --- | --- |
| DRAM | Dynamic Random Access Memory | 动态随机存取存储器 |
| 1T1C | 1 Transistor 1 Capacitor | 一个晶体管加一个电容的存储单元结构 |
| CD | Critical Dimension | 关键尺寸（线宽/孔径） |
| CDU | Critical Dimension Uniformity | 关键尺寸均匀性 |
| Overlay | — | 套刻（层间对准）精度 |
| FEOL | Front End Of Line | 前道工序（晶体管形成阶段） |
| BEOL | Back End Of Line | 后道工序（金属互连阶段） |
| CMP | Chemical Mechanical Polishing | 化学机械平坦化 |
| ALD | Atomic Layer Deposition | 原子层沉积 |
| CVD | Chemical Vapor Deposition | 化学气相沉积 |
| PVD | Physical Vapor Deposition | 物理气相沉积 |
| DOF | Depth of Focus | 光刻焦深 |
| High-k | High dielectric constant | 高介电常数材料 |
| Aspect Ratio | — | 深宽比 |

## 2. 测试与良率

| 缩写 | 全称 | 中文含义 |
| --- | --- | --- |
| WAT | Wafer Acceptance Test | 晶圆允收测试（电性参数测试） |
| PCM | Process Control Monitor | 工艺控制监控（与 WAT 常混用） |
| CP | Circuit Probe | 晶圆针测 |
| Wafer Sort | — | 晶圆分类测试（与 CP 常混用） |
| FT | Final Test | 封装后最终测试 |
| Bin | — | 分类代码，按测试结果给 die 分类 |
| Binning | — | 按测试结果将 die 分入不同 bin |
| Wafer Map | — | 晶圆图，标出各 die 的测试结果 |
| Yield | — | 良率 |
| Killer Defect | — | 致命缺陷（会导致 die 失效） |
| SPEC | Specification | 规格书，规定参数限值 |
| USL / LSL | Upper/Lower Specification Limit | 规格上下限 |

## 3. 工艺控制

| 缩写 | 全称 | 中文含义 |
| --- | --- | --- |
| APC | Advanced Process Control | 先进工艺控制 |
| R2R | Run-to-Run | 批次间控制（用上批结果调下批参数） |
| EWMA | Exponentially Weighted Moving Average | 指数加权移动平均 |
| dEWMA | double EWMA | 双 EWMA，用于带漂移的过程 |
| FDC | Fault Detection and Classification | 故障检测与分类 |
| VM | Virtual Metrology | 虚拟量测 |
| SPC | Statistical Process Control | 统计过程控制 |
| OOC | Out of Control | 过程失控（过程层面） |
| OOS | Out of Spec | 超出规格（产品层面） |
| Cpk / Ppk | Process Capability Index | 过程能力指数 |
| PM | Preventive Maintenance | 预防性维护 |
| MTBF | Mean Time Before Failure | 平均无故障时间 |
| MTTR | Mean Time To Repair | 平均修复时间 |
| OEE | Overall Equipment Effectiveness | 综合设备效率 |
| RUL | Remaining Useful Life | 剩余可用寿命 |
| PCA | Principal Component Analysis | 主成分分析 |
| SPE / Q | Squared Prediction Error | 平方预测误差（Q 统计量） |
| AMC | Airborne Molecular Contamination | 气态分子污染 |
| Recipe | — | 工艺配方（一套工艺参数的组合） |
| Step | — | 工艺步骤（recipe 中的一段） |
| Trace | — | 设备传感器记录的时序曲线 |
| Chamber | — | 腔体（工艺发生的封闭空间） |

## 4. 数据与算法

| 缩写 | 全称 | 中文含义 |
| --- | --- | --- |
| RCA | Root Cause Analysis | 根因分析 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| LLM | Large Language Model | 大语言模型 |
| MCP | Model Context Protocol | 模型上下文协议 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 |
| BM25 | Best Matching 25 | 一种经典关键词检索打分算法 |
| MRR | Mean Reciprocal Rank | 平均倒数排名 |
| nDCG | normalized Discounted Cumulative Gain | 归一化折损累计增益 |
| PR-AUC | Precision-Recall AUC | 精确率-召回率曲线下面积 |
| PSI | Population Stability Index | 群体稳定性指标 |
| SHAP | SHapley Additive exPlanations | 一种模型归因方法 |
| GroupKFold | — | 按组切分的交叉验证（防止同组泄漏） |
