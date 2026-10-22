# RAG Knowledge Base Agent

A document-QA agent with a **measured** retrieval-augmented generation pipeline — plus the
evaluation harness, the calibration work, and the engineering log that produced it.

Most RAG demos show a chat window. This one is organised around the harder question:
*how do you know it works, and where does it break?* Every number below is reproducible from
the scripts in this repository.

---

## Highlights

| Layer | Metric | This project | Notes |
| --- | --- | --- | --- |
| Retrieval | **Recall@3** (hybrid, two-stage) | **0.88 – 0.92** | 3 corpora, 98 questions |
| Retrieval | MRR | 0.83 | |
| Retrieval | vs. pure-vector / pure-BM25 | **0.84 / 0.75** | hybrid wins consistently |
| Generation | **Hallucination rate** | **0.0** | 18/18 out-of-corpus questions correctly refused |
| Generation | First-pass JSON validity | **1.0** | strict schema + repair loop |
| Latency | **TTFT** median / P95 | **1.8–2.1s / ≤ 4.2s** | after removing chain-of-thought |
| Latency | End-to-end median | **4.5–6.4s** | pooled over 294 samples |
| Correctness | LLM-judge score | 0.79 – 0.94 | calibrated against human labels, κ = 0.52 |

Three knowledge bases are indexed and switchable at runtime: DRAM manufacturing (47 chunks),
an AI/LLM interview-preparation corpus (229 chunks), and an Agent-evaluation handbook (37 chunks).

---

## Architecture

```
documents ──▶ parse ──▶ structure-aware chunking ──▶ embedding index (bge-m3, 1024-d)
                                                 └─▶ BM25 inverted index
                                                              │
query ──▶ two-stage retrieval ────────────────────────────────┘
            stage 1  wide recall:  Top-40 vector ∪ Top-40 BM25
            stage 2  weighted score-fusion rerank ──▶ Top-3
                                                       │
                                    ┌──────────────────┘
                                    ▼
                     grounded generation (strict JSON + citations)
                     validation → repair loop → structured answer
```

The whole agent is a **single 1,100-line Python file** with **no third-party dependencies
except NumPy** — BM25, Reciprocal Rank Fusion, the SSE parser and the HTTP client are
hand-written. That is deliberate: it keeps every stage explainable and the deliverable
runnable on any machine without a framework install.

---

## Quick start

```bash
# 1) API key — any OpenAI-compatible provider with both chat + embeddings
#    (tested with SiliconFlow: BAAI/bge-m3 + Qwen3-8B)
cp RAG知识库Agent/demo/.env.example RAG知识库Agent/demo/.env
#    then fill in LLM_BASE_URL / LLM_API_KEY / EMB_MODEL / LLM_MODEL

# 2) Build the index (or use --hash-embed for a fully offline, no-cost smoke test)
python RAG知识库Agent/demo/kb_agent.py ingest --docs docs --index index

# 3) Ask a question
python RAG知识库Agent/demo/kb_agent.py ask --index index \
  --q "ALM-1042 报警怎么处理？" --show-context

# 4) Or use the web UI (multi-corpus switching, streaming, clickable citations)
python RAG-WebUI/server.py            # http://127.0.0.1:8848
```

> Without an API key the pipeline still runs: `--hash-embed` swaps in an offline bag-of-words
> representation so the whole chain is exercisable at zero cost.

---

## Evaluation design

A benchmark is only as good as its question design, so the eval sets are split into four
categories that stress different failure modes:

| Category | What it stresses | Example |
| --- | --- | --- |
| **A** exact match | keyword/BM25 path | `ALM-1042 报警的触发条件是什么？` |
| **B** semantic generalisation | vector path (no lexical overlap) | `怎么让工艺在批次之间保持稳定？` |
| **C** multi-chunk aggregation | recall breadth | `APC 的三层分别负责什么？` |
| **D** unanswerable | **hallucination** | `EUV 光刻的曝光波长是多少纳米？` |

Category D is the important one: it deliberately asks things the corpus cannot answer
(device models, financials, alarm codes that do not exist) so that *refusing* is testable
rather than assumed. The measured hallucination rate of 0.0 comes from these cases.

Every eval set ships with `gold_cids` and a reference answer, and the builder validates both
(`build_eval.py`) — including an entity-consistency check that catches mislabelled gold chunks,
the failure mode that silently depresses recall without raising an error.

---

## Engineering findings

These are the parts worth reading. The full log is in [`变更记录.md`](变更记录.md) (Chinese).

**1. The real latency bottleneck was chain-of-thought, not output length.**
The first token arrived at 17.8 s while the whole answer took 23 s — the model was emitting
400+ reasoning chunks before any content. Disabling thinking mode cut TTFT to 0.53 s.
TTFT was then promoted to a first-class metric.

**2. Single-run P95 latency is noise.**
A 3-run × 3-corpus experiment (294 generations) showed per-run P95 jitters 30–68 %, and even
the *median* moved up to 51 % between runs — the API itself gets slower in bursts. Only
quantiles pooled across runs are trustworthy; the apparent "one corpus is slower" result
disappeared once pooling was applied.

**3. Two-stage retrieval beat baseline fusion, and the reason is measurable.**
Recall rose from 0.8387 to 0.8710 when RRF was replaced by score-normalised weighted fusion
(α = 0.7, selected by 5-fold cross-validation, all five folds converging independently).
RRF discards score magnitude — it cannot distinguish a clear #1 from a marginal one.

**4. Text chunking produced three real bugs, all found by measuring.**
Sentence-boundary cutting interacted badly with overlap: when a chunk is shorter than the
overlap, `end - overlap` walks backwards and shreds a paragraph into dozens of 1-character
fragments. Cutting inside brackets produced duplicated spans. Both are documented and fixed,
and the fix is scoped to an opt-in path so the published index and its metrics do not drift.

**5. An LLM judge must be calibrated, and calibration exposed its own limits.**
The correctness scorer agreed with human labels at κ = 0.52 — a *moderate* result. It was
validated on a purpose-built hard suite (10 off-topic answers, 10 paraphrased-correct answers)
where it scored 1.00 on both, proving it catches gross irrelevance and does not penalise
paraphrase — but it still under-penalises answers that are mostly right with one inverted fact.
The tooling for a two-annotator IAA study ships here (`annotate.html`, `iaa_calc.py`).

---

## Repository layout

```
RAG知识库Agent/
├── demo/
│   ├── kb_agent.py          # the entire agent: ingest / ask / eval / compare
│   ├── docs/                # DRAM manufacturing corpus (12 documents)
│   ├── eval/                # 38-question eval set + machine-readable reports
│   └── README.md            # design trade-offs and measured results
└── 评测与报告/                # written evaluation report (DOCX) + raw results

RAG-WebUI/
├── server.py                # zero-dependency HTTP server (multi-corpus, NDJSON streaming)
├── index.html               # web UI: corpus switcher, streaming, citation → chunk mapping
├── annotate.html            # annotation UI for human-labelling studies
├── annotation_server.py
├── build_index.py           # registry-driven index builder
├── build_eval.py            # eval-set builder with gold-chunk validation
├── run_eval.py              # retrieval + generation metrics
├── accuracy_eval.py         # LLM-as-judge correctness scoring
├── judge_calibration.py     # Cohen's κ against human labels
├── build_hard_suite.py      # adversarial judge test suite
├── judge_hard_eval.py
├── latency_repeat.py        # repeated-run latency study
├── iaa_calc.py              # inter-annotator agreement
└── kb/
    ├── registry.json        # corpus definitions (docs/index paths, chunking params)
    └── */docs/              # three switchable corpora
```

---

## Notes

- **The corpora are illustrative, not proprietary.** The DRAM documents were written for this
  project from public semiconductor-manufacturing fundamentals. No production or company data
  is included.
- Chinese folder names in the tree are referenced by path in the code; renaming them requires
  updating the `DEMO` path constants.
- Indexes are not committed — regenerate them with `build_index.py` (or `ingest`).
