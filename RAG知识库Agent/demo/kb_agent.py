# -*- coding: utf-8 -*-
"""
kb_agent —— 零第三方依赖的「文档知识库 Agent」（用于长鑫 J20191 面试 Demo）
================================================================================
设计取舍（面试可以直接讲）：
  1) 零第三方依赖：标准库 + numpy。任何机器 clone 下来配好 key 就能跑，
     不会因为 pip/网络/版本问题在面试前卡住。工程上这叫「降低交付摩擦」。
  2) 自实现 BM25 与 RRF：不引 rank_bm25 / 不依赖向量库，
     因为要能解释每一步内部在做什么（面试官会追问公式）。
  3) 混合检索 = 向量(语义) + BM25(精确) → RRF 融合 → 可选 rerank。
     理由：专有名词/型号/报警码这类精确匹配，纯向量会糊掉。
  4) 生成侧强制「带引用、无依据就说不知道」，并做结构化输出校验。
     这是抑制幻觉最有效的一招，也是可评测的前提。
  5) 评测与主程序解耦：eval 子命令读 eval_set.jsonl，输出结构化报告。
     对应 JD 职责第 2 条「构建评估数据集，输出评估报告」。

用法：
  python kb_agent.py ingest  --docs docs/ --index index/
  python kb_agent.py ask     --index index/ --q "你的问题" [--show-context]
  python kb_agent.py eval    --index index/ --set eval/eval_set.jsonl --out eval/report.json
  python kb_agent.py compare --index index/ --set eval/eval_set.jsonl   # 向量 vs BM25 vs 混合

环境变量：
  LLM_BASE_URL   默认 https://api.openai.com/v1   （DeepSeek 用 https://api.deepseek.com/v1）
  LLM_API_KEY    必填
  EMB_MODEL      默认 text-embedding-3-small      （不兼容时可换成本地词袋，见 --hash-embed）
  LLM_MODEL      默认 gpt-4o-mini                 （DeepSeek 用 deepseek-chat）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 0. 配置
#
# API key 的配置方式（优先级从高到低）：
#   1) 终端的真实环境变量          $env:LLM_API_KEY="sk-..."
#   2) 项目根目录的 .env 文件（推荐；下面会自动加载，不用每次重设）
#   3) 两个都没有 → 自动降级为「只跑检索层评测」，不报错（见 evaluate()）
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> int:
    """极简 .env 加载器（零依赖）。已存在的真实环境变量优先，不会被覆盖。"""
    if not path.exists():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.split(" #")[0].strip()          # 去掉行尾注释
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]                        # 去掉包裹的引号
        if key and key not in os.environ:          # 真实环境变量优先
            os.environ[key] = val
            count += 1
    return count


_loaded = load_dotenv(BASE_DIR / ".env")

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
USE_HASH_EMBED = os.environ.get("USE_HASH_EMBED", "").lower() in {"1", "true", "yes"}

# 混合推理模型（Qwen3 等）默认会先输出一大段思维链再给正文，
# 实测 8B 模型正文首块要等约 17 秒，而关掉后 0.5 秒就开始出字。
# 本项目不需要思维链（答案受限于给定资料，推理空间很小），
# 所以默认关闭；要打开就设 DISABLE_THINKING=0。
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "1").lower() not in {"0", "false", "no"}


def thinking_kwargs() -> dict:
    """给请求体加上关闭思维链的参数（不同厂商参数名不一致，这里按需生效）。

    未知参数会被服务端忽略，所以不支持的平台不会因此报错。
    """
    if not DISABLE_THINKING:
        return {}
    return {"enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none"}

# 中文/英文混排切词：中文按字 + 英文数字按词。够用且零依赖。
TOKEN_RE = re.compile(r"[A-Za-z0-9_./\-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ----------------------------------------------------------------------------
# 1. HTTP：只依赖标准库，方便在任何机器上跑
# ----------------------------------------------------------------------------
def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:500]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络不可达 {url}: {e.reason}") from e


def embed_texts(texts: list[str], batch: int = 64, retries: int = 3) -> np.ndarray:
    """调 embedding API，返回 (n, d) 的 float32 矩阵，已做 L2 归一化。

    兼容两种返回格式，因为不同厂商实现不一致：
      A) OpenAI 官方风格： {"data": [{"index": 0, "embedding": [...]}, ...]}
      B) 简化风格：        {"data": [[...], [...]]}   （部分国内厂商如此）
    """
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        last_err = None
        for attempt in range(retries):
            try:
                resp = _post_json(f"{BASE_URL}/embeddings",
                                  {"model": EMB_MODEL, "input": chunk})
                data = resp["data"]
                if data and isinstance(data[0], dict):       # 格式 A
                    items = sorted(data, key=lambda x: x["index"])
                    out.extend([it["embedding"] for it in items])
                else:                                        # 格式 B
                    out.extend([list(map(float, v)) for v in data])
                last_err = None
                break
            except Exception as e:                      # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)                # 指数退避
        if last_err:
            raise RuntimeError(f"embedding 失败（已重试 {retries} 次）: {last_err}")
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def chat(messages: list[dict], temperature: float = 0.0, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            payload = {"model": LLM_MODEL, "messages": messages,
                       "temperature": temperature}
            payload.update(thinking_kwargs())
            resp = _post_json(f"{BASE_URL}/chat/completions", payload)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:                          # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat 失败（已重试 {retries} 次）: {last_err}")


def chat_stream(messages: list[dict], temperature: float = 0.0, retries: int = 3,
                with_reasoning: bool = False):
    """流式版 chat：逐块 yield (类型, 文本)。

    类型为 "reasoning"（思维链）或 "content"（正文）。
    with_reasoning=True 时才吐思维链——它不算答案，只是用来安抚等待，
    但 Qwen3 这类混合推理模型在正文前会先输出 400+ 块思维链，
    不吐出来的话界面上会干等十几秒，所以值得单独给一条通道。
    """
    payload = {"model": LLM_MODEL, "messages": messages,
               "temperature": temperature, "stream": True}
    payload.update(thinking_kwargs())
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {API_KEY}"},
                method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        return
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if with_reasoning and delta.get("reasoning_content"):
                        yield ("reasoning", delta["reasoning_content"])
                    if delta.get("content"):
                        yield ("content", delta["content"])
            return
        except Exception as e:                          # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"流式 chat 失败（已重试 {retries} 次）: {last_err}")


# ----------------------------------------------------------------------------
# 2. 离线 hash embedding（不联网的备选，用于对比实验 / API 挂了的降级）
# ----------------------------------------------------------------------------
def hash_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    """把 token 哈希到固定维度并做 L2 归一化。不是好模型，但是个诚实的基线。"""
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            mat[i, h % dim] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


# ----------------------------------------------------------------------------
# 3. 文档解析与切分
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    cid: str
    doc: str
    text: str
    meta: dict = field(default_factory=dict)


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        # 零依赖：提示用户先转文本，并给出具体命令（不要静默失败）
        raise RuntimeError(
            f"PDF 需要先转文本：{path.name}\n"
            f"  推荐：pdftotext -layout \"{path}\" \"{path.with_suffix('.txt')}\"\n"
            f"  或：python -m pip install pypdf 后自行扩展 read_file"
        )
    if suffix == ".docx":
        from docx import Document as _Doc          # python-docx 可用
        d = _Doc(str(path))
        parts = [p.text for p in d.paragraphs]
        for tb in d.tables:                         # 表格必须单独处理，否则检索不到
            for row in tb.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts)
    raise RuntimeError(f"不支持的格式：{suffix}")


_OPEN_BRACKETS = "（(【[「“"
_CLOSE_BRACKETS = "）)】]」”"


def _cut_points(s: str, size: int) -> int:
    """返回切点：尽量在 size 处、且不落在括号内部的标点之后。

    设计说明（这里踩过两个坑，都写在注释里）：
      1) 不能在括号内部切。切在「（本地 CPU/消费级」的逗号上会把括号短语劈开，
         再加重叠拼接就产生重复文本。
      2) 不能在 size 之前「回看」找标点。切点越靠近 size，下一轮
         start = end - overlap 反而越小，配合重叠会退化成每次只前进 1 个字符，
         无限切碎。所以只在 [size-40, size] 这个极窄窗口内找标点，
         找不到就直接硬切 size——保证每轮至少前进 size - overlap。
    """
    hi = min(size, len(s))
    if hi <= 0:
        return size
    lo = max(1, hi - 40)

    # 前缀深度：depth[i] = s[:i] 的净括号深度，depth[k]==0 即可安全切在 k 处
    depth = [0] * (hi + 1)
    d = 0
    for i in range(hi):
        ch = s[i]
        if ch in _OPEN_BRACKETS:
            d += 1
        elif ch in _CLOSE_BRACKETS:
            d = max(0, d - 1)
        depth[i + 1] = d

    for klass in ("。！？!?；\n", ",，;；"):
        for k in range(hi, lo - 1, -1):
            if depth[k] == 0 and s[k - 1] in klass:
                return k
    return hi


def _merge_overlap(chunks: list[str], overlap: int) -> list[str]:
    """把上一块的尾部拼到当前块前面，避免答案被跨块切断。"""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    merged = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        merged.append((prev[-overlap:] + "\n" + cur).strip())
    return merged


def _tail_overlap(prev: str, cur: str, overlap: int) -> str:
    """给 cur 取一段仍然携带上下文的 prev 尾部。

    直接切固定长度会出问题：prev 末尾若是重复的表格行或被上一块覆盖过的内容，
    拼进来就变成「测试团队的工作，\\n作，\\n，是所有人的工作」这种碎片。
    所以从 75 字开始往回缩，取第一个「不整段出现在 cur 里」的后缀。
    """
    for ln in range(overlap, 0, -1):
        tail = prev[-ln:]
        if tail.strip() and tail not in cur:
            return tail
    return ""


def _trim_seam(tail: str, cur: str, max_trim: int = 12) -> tuple[str, str]:
    """裁掉 tail 末尾与 cur 开头的重叠部分。

    tail 是从 prev 尾部切出来的，视角不同，切点可能落在词中间
    （如 prev 结尾是「…很多向量库为了性能用内积。」而 tail 开头是「用。」），
    直接拼上去就会固化半个词。所以检测真实重叠并按较长的那个方向裁。
    """
    if not tail or not cur:
        return tail, cur
    limit = min(len(tail), len(cur), max_trim)
    for k in range(limit, 1, -1):
        if tail.endswith(cur[:k]):
            return tail[:-k], cur
        if cur.startswith(tail[-k:]):
            return tail, cur[k:]
    return tail, cur


def _merge_overlap_clean(chunks: list[str], overlap: int) -> list[str]:
    """句感知路径用：加缝时避免把重复内容固化进 chunk。"""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    merged = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = _tail_overlap(prev, cur, overlap)
        tail, cur = _trim_seam(tail, cur)
        merged.append(f"{tail}\n{cur}".strip() if tail else cur)
    return merged


def _split_paragraph_sentence_aware(p: str, size: int, overlap: int) -> list[str]:
    """超长段落切分（句感知版）。

    与硬切版的区别，都是为了质量：
      1) 尽量切在句末/逗号，且不落在括号内部；
      2) 重叠不得超过本块长度的 1/3，而且切点比 overlap 还小时干脆不重叠——
         加重叠的前提是「步进 = 块长 - overlap」仍然为正，否则 end - ov 会退回
         start 附近，每轮只前进 1 个字符，把一个段落切成几十个碎片；
      3) 过短的片（含段落尾巴）并入前一片，避免留下「作，」「，是所有人的工作 |」
         这类碎片污染向量空间。
    """
    pieces: list[str] = []
    start = 0
    while start < len(p):
        cut = _cut_points(p[start:start + size], size)
        end = start + cut
        piece = p[start:end]
        if len(piece) < 20 and pieces:
            pieces[-1] += piece                    # 太短，直接并进前一片
        else:
            pieces.append(piece)
        if end >= len(p):
            break
        # 与本片长度成比例，且步进必须为正
        ov = min(overlap, max(1, len(piece) // 3)) if cut > overlap else 0
        start = max(end - ov, start + 1)
    return pieces


def chunk_text(text: str, size: int = 500, overlap: int = 75,
               sentence_aware: bool = False) -> list[str]:
    """按段落聚合到目标长度，保留结构；段落过长再切。overlap 按字符数。

    sentence_aware=False（默认）保持与原实现完全一致的硬切行为，
    这样已发布索引与评测指标不会漂移；True 才启用句边界切分，
    适合长段落密集的文档（如备考清单）。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if sentence_aware:
        chunks: list[str] = []
        buf = ""
        for p in paras:
            if len(p) <= size:
                if len(buf) + len(p) + 1 <= size:
                    buf = f"{buf}\n{p}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = p
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            for piece in _split_paragraph_sentence_aware(p, size, overlap):
                if len(buf) + len(piece) + 1 <= size:
                    buf = f"{buf}\n{piece}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = piece
        if buf:
            chunks.append(buf)
        return _merge_overlap_clean(chunks, overlap)

    # ---- 原始硬切路径（勿改，见 docstring）----
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}".strip()
            continue
        if buf:
            chunks.append(buf)
        if len(p) <= size:
            buf = p
        else:                                       # 超长段落硬切 + 重叠
            start = 0
            while start < len(p):
                chunks.append(p[start:start + size])
                start += size - overlap
            buf = ""
    if buf:
        chunks.append(buf)
    return _merge_overlap(chunks, overlap)


def ingest(docs_dir: Path, index_dir: Path, use_hash: bool = False,
           chunk_size: int = 500, overlap: int = 75,
           sentence_aware: bool = False) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = read_file(path)
        except RuntimeError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        for i, piece in enumerate(chunk_text(text, size=chunk_size, overlap=overlap,
                                             sentence_aware=sentence_aware)):
            chunks.append(Chunk(cid=f"{path.stem}#{i}", doc=path.name, text=piece))
    if not chunks:
        raise SystemExit("没有解析到任何文档，请检查 docs/ 目录")
    print(f"切分完成：{len(chunks)} 个 chunk，来自 "
          f"{len({c.doc for c in chunks})} 个文档")

    vectors = (hash_embed([c.text for c in chunks]) if use_hash
               else embed_texts([c.text for c in chunks]))
    np.save(index_dir / "vectors.npy", vectors)
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1),
        encoding="utf-8")
    (index_dir / "meta.json").write_text(json.dumps({
        "embed_model": "hash-512" if use_hash else EMB_MODEL,
        "n_chunks": len(chunks), "dim": int(vectors.shape[1]),
        "chunk_size": chunk_size, "overlap": overlap,
        "sentence_aware": sentence_aware,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"索引已写入 {index_dir}（维度 {vectors.shape[1]}）")


# ----------------------------------------------------------------------------
# 4. 检索：BM25 + 向量 + RRF
# ----------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.n = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.n, 1)
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in tokenize(query):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if not f:
                    continue
                dl = len(self.docs[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion：用排名而非分数融合，免去量纲/权重调参。"""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos)
    return fused


RERANK_MODES = ("rrf", "max", "minmax", "lex")


def fuse_scores(vs: np.ndarray, bs: np.ndarray, cand: list[int],
                mode: str = "rrf", alpha: float = 0.5, rrf_k: int = 60,
                query: str = "", chunks: list[dict] | None = None,
                lex_w: float = 0.15) -> dict[int, float]:
    """把候选集按指定策略重排，返回 {index: score}。

    设计说明（面试可讲）：这是「召回-重排」两阶段里的第二阶段。
    第一阶段用宽召回保证答案进入候选池（实验显示 Top-8 即可达到 Recall 1.0），
    第二阶段只需在候选池内把正确的挑到最前面。

    策略：
      rrf    : 归一化 RRF 融合分（只用排名，无法区分「第1名远超第2名」）
      max    : 各路分数按最大值归一化后加权 —— 保留分数间的相对落差
      minmax : 各路分数按 min-max 归一化后加权 —— 放大池内差异
      lex    : max 加权融合 + 词面覆盖率加分 —— 引入第三种信号
    """
    if mode == "rrf":
        v_rank = list(np.argsort(-vs))
        b_rank = list(np.argsort(-bs))
        # 只保留候选集内的元素
        v_rank = [i for i in v_rank if i in set(cand)]
        b_rank = [i for i in b_rank if i in set(cand)]
        fused = rrf([v_rank, b_rank], k=rrf_k)
        top = max(fused.values()) if fused else 1.0
        return {i: fused.get(i, 0.0) / top for i in cand}

    cv, cb = vs[cand], bs[cand]

    def norm(x, how):
        if how == "max":
            m = float(np.max(x)) if len(x) else 0.0
            return x / m if m > 0 else np.zeros_like(x)
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.ones_like(x)
        return (x - lo) / (hi - lo)

    how = "minmax" if mode == "minmax" else "max"
    nv, nb = norm(cv, how), norm(cb, how)
    score = alpha * nv + (1 - alpha) * nb

    if mode == "lex" and query:
        q_tokens = set(tokenize(query))
        if q_tokens:
            cov = np.array([
                len(q_tokens & set(tokenize(chunks[i]["text"]))) / len(q_tokens)
                for i in cand
            ], dtype=np.float32)
            score = score + lex_w * cov

    return {i: float(score[j]) for j, i in enumerate(cand)}


class Retriever:
    def __init__(self, index_dir: Path):
        self.chunks = json.loads((index_dir / "chunks.json").read_text("utf-8"))
        self.vectors = np.load(index_dir / "vectors.npy")
        self.meta = json.loads((index_dir / "meta.json").read_text("utf-8"))
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])

    def _vec_scores(self, query: str) -> np.ndarray:
        if self.meta["embed_model"] == "hash-512":
            qv = hash_embed([query])[0]
        else:
            qv = embed_texts([query])[0]
        return self.vectors @ qv                       # 已归一化 → 点积即余弦

    def candidates(self, query: str, pool: int = 10) -> tuple[list[int], np.ndarray, np.ndarray]:
        """第一阶段：宽召回。返回 (候选 index 列表, 向量分, BM25 分)。

        候选集 = 向量 Top-pool ∪ BM25 Top-pool。
        取并集而不是只看某一路，是为了让两路各自的独有命中都能进入重排阶段。
        """
        vs = self._vec_scores(query)
        bs = self.bm25.scores(query)
        v_top = set(int(i) for i in np.argsort(-vs)[:pool])
        b_top = set(int(i) for i in np.argsort(-bs)[:pool])
        cand = sorted(v_top | b_top)
        return cand, vs, bs

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid",
               pool: int = 30, rerank: str | None = None,
               rerank_pool: int = 10, alpha: float = 0.5,
               lex_w: float = 0.15) -> list[tuple[int, float]]:
        vs = self._vec_scores(query)
        bs = self.bm25.scores(query)
        if mode == "vector":
            order = np.argsort(-vs)[:top_k]
            return [(int(i), float(vs[i])) for i in order]
        if mode == "bm25":
            order = np.argsort(-bs)[:top_k]
            return [(int(i), float(bs[i])) for i in order]

        if rerank:
            # 两阶段：宽召回 → 重排
            cand, _, _ = self.candidates(query, pool=rerank_pool)
            fused = fuse_scores(vs, bs, cand, mode=rerank, alpha=alpha,
                                query=query, chunks=self.chunks, lex_w=lex_w)
            best = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
            return [(int(i), float(s)) for i, s in best]

        v_rank = list(np.argsort(-vs)[:pool])
        b_rank = list(np.argsort(-bs)[:pool])
        fused = rrf([v_rank, b_rank])
        best = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return [(int(i), float(s)) for i, s in best]


# ----------------------------------------------------------------------------
# 5. 生成：强制引用 + 结构化输出
# ----------------------------------------------------------------------------
SYS_PROMPT = """你是面向产线工程师的设备与工艺知识助手。

硬性规则：
1. 只依据 <context> 中的资料回答，不得使用资料外的知识。
2. 每个事实性结论后面必须标注来源编号，格式如 [1]、[2]。
3. 如果资料不足以回答，必须把 "answerable" 设为 false，answer 写"资料不足"，
   并在 missing 里说明缺什么。绝对不要猜测。
4. 只输出一个 JSON 对象，不要输出 Markdown 代码块或任何解释。

输出 JSON Schema：
{"answerable": bool, "answer": str, "citations": [int], "missing": str}
"""


def build_context(hits: list[tuple[int, float]], chunks: list[dict]) -> str:
    lines = []
    for n, (idx, _score) in enumerate(hits, start=1):
        c = chunks[idx]
        lines.append(f"[{n}] 来源：{c['doc']} / {c['cid']}\n{c['text']}")
    return "<context>\n" + "\n\n".join(lines) + "\n</context>"


def parse_json_lenient(text: str) -> tuple[dict | None, str | None]:
    """结构化输出校验：失败时返回错误原因，供上层做「回退重修」。"""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e.msg}"
    if not isinstance(obj, dict):
        return None, "输出不是 JSON 对象"
    for key, typ in (("answerable", bool), ("answer", str),
                     ("citations", list), ("missing", str)):
        if key not in obj:
            return None, f"缺少字段 {key}"
        if not isinstance(obj[key], typ):
            return None, f"字段 {key} 类型应为 {typ.__name__}"
    if obj["answerable"] and not obj["citations"]:
        return None, "answerable=true 但没有任何引用"
    return obj, None


def answer(query: str, retriever: Retriever, top_k: int = 5,
           mode: str = "hybrid", max_repair: int = 2,
           rerank: str | None = None, rerank_pool: int = 40,
           alpha: float = 0.7) -> dict:
    hits = retriever.search(query, top_k=top_k, mode=mode, rerank=rerank,
                            rerank_pool=rerank_pool, alpha=alpha)
    ctx = build_context(hits, retriever.chunks)
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"{ctx}\n\n问题：{query}"}]

    repairs = 0
    while True:
        raw = chat(messages)
        obj, err = parse_json_lenient(raw)
        if obj is not None:
            obj["_repairs"] = repairs
            obj["_hits"] = [retriever.chunks[i]["cid"] for i, _ in hits]
            obj["_raw"] = raw
            return obj
        # 回退重修：把具体校验错误喂回去让它自己改
        if repairs >= max_repair:
            return {"answerable": False, "answer": "结构化输出校验失败，已降级",
                    "citations": [], "missing": err or "未知",
                    "_repairs": repairs, "_hits": [retriever.chunks[i]["cid"]
                                                   for i, _ in hits],
                    "_raw": raw, "_failed": True}
        repairs += 1
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": f"你的输出不合法：{err}。请只输出合法 JSON。"})


def ask_stream(query: str, retriever: "Retriever", top_k: int = 5,
               mode: str = "hybrid", max_repair: int = 2,
               rerank: str | None = None, rerank_pool: int = 40,
               alpha: float = 0.7):
    """流式版 answer()：yield ("delta", 文本增量) / 最后 yield ("result", 完整 dict)。

    为什么要边流边解析：系统提示强制输出 JSON，
    期望形态是 {"answerable": true, "answer": "...", "citations": [1]}。
    如果原样把 JSON 吐给用户，界面上会出现一堆引号和花括号。
    所以这里用「提取 answer 字段值」的做法，只把 answer 的内容流出去；
    校验仍然在流结束后完整做一遍（和 answer() 同一套规则），
    失败则把具体错误喂回去重修，重修时前端会收到 reset 事件清空重来。
    """
    hits = retriever.search(query, top_k=top_k, mode=mode, rerank=rerank,
                            rerank_pool=rerank_pool, alpha=alpha)
    ctx = build_context(hits, retriever.chunks)
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"{ctx}\n\n问题：{query}"}]

    repairs = 0
    while True:
        buf: list[str] = []
        emitted = 0
        extractor = _AnswerExtractor()
        for kind, piece in chat_stream(messages, with_reasoning=True):
            if kind == "reasoning":
                # 思维链单独一路吐出去，只为缓解等待焦虑，不参与解析
                yield ("reasoning", piece)
                continue
            buf.append(piece)
            got = extractor.feed("".join(buf))
            if got:
                emitted += len(got)
                yield ("delta", got)

        raw = "".join(buf)
        obj, err = parse_json_lenient(raw)
        if obj is not None:
            obj["_repairs"] = repairs
            obj["_hits"] = [retriever.chunks[i]["cid"] for i, _ in hits]
            obj["_raw"] = raw
            yield ("result", obj)
            return

        if repairs >= max_repair:
            yield ("result", {"answerable": False,
                              "answer": "结构化输出校验失败，已降级",
                              "citations": [], "missing": err or "未知",
                              "_repairs": repairs,
                              "_hits": [retriever.chunks[i]["cid"] for i, _ in hits],
                              "_raw": raw, "_failed": True})
            return
        repairs += 1
        yield ("reset", err)                       # 让前端清掉上一次的半截输出
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": f"你的输出不合法：{err}。请只输出合法 JSON。"})


_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\",
                 "/": "/", "b": "\b", "f": "\f"}


def _unescape_json(raw: str) -> str:
    """解开 JSON 字符串转义。raw 一定是完整转义（_scan 保证不在转义中间切断）。"""
    if "\\" not in raw:
        return raw
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch != "\\" or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        e = raw[i + 1]
        if e == "u":
            hexs = raw[i + 2:i + 6]
            try:
                out.append(chr(int(hexs, 16)))
            except ValueError:
                out.append("\\u" + hexs)
            i += 6
            continue
        out.append(_JSON_ESCAPES.get(e, "\\" + e))
        i += 2
    return "".join(out)


class _AnswerExtractor:
    """从逐步增长的 JSON 文本里增量提取 "answer" 字段的值。

    调用方每次传入「到目前为止的完整文本」，本类记录已产出的长度，
    只返回新增的部分。文本总量是几百字到几千字，重复扫描的开销可以忽略；
    换来的好处是状态极少、没有窗口对齐问题。
    """

    _KEY = '"answer"'

    def __init__(self) -> None:
        self.emitted = 0                 # 已产出的字符数
        self.started = False             # 是否已找到 answer 字段的值起点

    def feed(self, text: str) -> str:
        if not self.started and not self._locate(text):
            return ""
        i = self._scan(text, self._value_start)
        if i > self.emitted:
            raw = text[self._value_start + self.emitted:self._value_start + i]
            self.emitted = i
            return _unescape_json(raw)
        return ""

    def _locate(self, text: str) -> bool:
        pos = text.find(self._KEY)
        if pos < 0:
            return False
        colon = text.find(":", pos + len(self._KEY))
        if colon < 0:
            return False
        q = text.find('"', colon + 1)    # 值的起始引号
        if q < 0:
            return False
        self._value_start = q + 1
        self.started = True
        return True

    def _scan(self, text: str, start: int) -> int:
        """扫描值，返回「已确定完整处理」的字符数（相对于值起点）。"""
        i = self.emitted                # 已产出的部分不必重扫
        n = len(text) - start
        while i < n:
            ch = text[start + i]
            if ch == "\\":
                e = text[start + i + 1:start + i + 2]
                if not e:
                    break                        # 只见到 '\'，还不知道是哪种转义
                if e == "u":
                    hexs = text[start + i + 2:start + i + 6]
                    if len(hexs) < 4:
                        break                    # \uXXXX 没流完，等下一块
                    i += 6
                    continue
                i += 2
                continue
            if ch == '"':                        # 值的结束引号
                return i
            i += 1
        return i



# ----------------------------------------------------------------------------
# 6. 评测（对应 JD 职责第 2 条）
# ----------------------------------------------------------------------------
def load_eval_set(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def evaluate(index_dir: Path, eval_path: Path, out_path: Path,
             top_k: int = 5, modes: list[str] | None = None,
             retrieval_only: bool = False, rerank: str | None = None,
             rerank_pool: int = 40, alpha: float = 0.7) -> dict:
    retriever = Retriever(index_dir)
    cases = load_eval_set(eval_path)
    modes = modes or ["hybrid"]
    report: dict = {"n_cases": len(cases), "top_k": top_k, "modes": {},
                    "retrieval_only": retrieval_only,
                    "rerank": rerank, "rerank_pool": rerank_pool if rerank else None,
                    "alpha": alpha if rerank else None}

    # ---- 检索层指标（对每种检索模式都算，用于选型） ----
    for mode in modes:
        hit = 0
        rr_sum = 0.0
        for case in cases:
            gold = set(case.get("gold_cids", []))
            if not gold:
                continue
            hits = retriever.search(case["q"], top_k=top_k, mode=mode,
                                    rerank=rerank, rerank_pool=rerank_pool,
                                    alpha=alpha)
            got = [retriever.chunks[i]["cid"] for i, _ in hits]
            if gold & set(got):
                hit += 1
                rr_sum += 1.0 / (1 + min(got.index(g) for g in gold if g in got))
        n = sum(1 for c in cases if c.get("gold_cids"))
        report["modes"][mode] = {
            f"recall@{top_k}": round(hit / n, 4) if n else None,
            "mrr": round(rr_sum / n, 4) if n else None,
            "n_gold_cases": n,
        }

    # ---- 生成层指标（只对选了模式跑，省 token） ----
    if retrieval_only or not API_KEY:
        if not API_KEY:
            print("[warn] 未设置 LLM_API_KEY，跳过生成层评测，只输出检索层指标。",
                  file=sys.stderr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        return report

    gen_mode = modes[0]
    n_ans = n_tool_ok = n_json_ok = n_repair_total = 0
    n_answered_when_unanswerable = 0
    n_unanswerable_cases = 0
    latencies: list[float] = []
    rows = []
    for case in cases:
        t0 = time.time()
        res = answer(case["q"], retriever, top_k=top_k, mode=gen_mode,
                     rerank=rerank, rerank_pool=rerank_pool, alpha=alpha)
        dt = time.time() - t0
        latencies.append(dt)
        n_json_ok += 0 if res.get("_failed") else 1
        n_repair_total += res.get("_repairs", 0)
        gold = set(case.get("gold_cids", []))
        cited_hit = bool(gold & set(res.get("_hits", [])))
        if case.get("answerable", True):
            n_ans += 1
            n_tool_ok += 1 if res.get("answerable") else 0
        else:
            n_unanswerable_cases += 1
            if res.get("answerable"):
                n_answered_when_unanswerable += 1
        rows.append({"q": case["q"], "answerable": res.get("answerable"),
                     "answer": res.get("answer"), "citations": res.get("citations"),
                     "hits": res.get("_hits"), "gold": sorted(gold),
                     "retrieval_hit": cited_hit, "repairs": res.get("_repairs"),
                     "latency_s": round(dt, 2), "failed": res.get("_failed", False)})

    n = len(cases)
    report["generation"] = {
        "json_valid_rate": round(n_json_ok / n, 4) if n else None,
        "answerable_accuracy": round(n_tool_ok / n_ans, 4) if n_ans else None,
        "false_answer_rate_on_unanswerable":
            round(n_answered_when_unanswerable / n_unanswerable_cases, 4)
            if n_unanswerable_cases else None,
        "avg_repairs": round(n_repair_total / n, 3) if n else None,
        "p95_latency_s": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
    }
    report["cases"] = rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return report


def print_report(rep: dict) -> None:
    print("\n=== 检索层（不同检索模式对比）===")
    print(f"{'mode':10} {'Recall@k':>10} {'MRR':>8}  n_gold")
    for mode, m in rep["modes"].items():
        print(f"{mode:10} {str(m['recall@' + str(rep['top_k']) ]):>10} "
              f"{str(m['mrr']):>8}  {m['n_gold_cases']}")
    g = rep.get("generation")
    if not g:
        print("\n(仅检索层：未跑生成层评测)")
        return
    print("\n=== 生成层 ===")
    for k, v in g.items():
        print(f"  {k:38} {v}")
    # 区分三类 case，避免把「正确拒绝」误报成问题：
    #   1) should_refuse 且确实拒答了  → 正确行为，不算问题
    #   2) should_refuse 但答了        → 幻觉，必须关注
    #   3) 有 gold 但没检索到 / 输出不合法 → 真问题，必须关注
    should_refuse_ok, bad = [], []
    for c in rep["cases"]:
        has_gold = bool(c.get("gold"))
        refused = (c.get("answerable") is False)
        if not has_gold and refused and not c.get("failed"):
            should_refuse_ok.append(c)
        elif c.get("failed") or (has_gold and not c.get("retrieval_hit")):
            bad.append(c)
        elif not has_gold and not refused:
            bad.append(c)                     # 幻觉：该拒绝却回答了
    print(f"\n=== 正确拒绝（资料不足，符合预期）：{len(should_refuse_ok)} 条 ===")
    for c in should_refuse_ok:
        print(f"  ✓ {c['q'][:46]!r}")
    print(f"\n=== 需要关注的 case：{len(bad)} / {len(rep['cases'])} ===")
    for c in bad[:8]:
        print(f"  - {c['q'][:46]!r}  hit={c['retrieval_hit']} "
              f"failed={c['failed']} repairs={c['repairs']}")


# ----------------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="零依赖文档知识库 Agent（面试 Demo）")
    ap.add_argument("--hash-embed", action="store_true", default=None,
                    help="用离线 hash embedding，不调 API（做对比实验用）。"
                         "也可用 .env 里的 USE_HASH_EMBED=1 开启")
    ap.add_argument("--rerank", default="max", choices=["off", *RERANK_MODES],
                    help="重排策略。默认 max；传 off 关闭并退回原 RRF 基线")
    ap.add_argument("--rerank-pool", type=int, default=40,
                    help="重排前的候选池大小（两路并集）")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="向量分权重（0~1）。经 5 折交叉验证选得 0.7")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="解析文档并建索引")
    p_ing.add_argument("--docs", default="docs")
    p_ing.add_argument("--index", default="index")
    p_ing.add_argument("--chunk-size", type=int, default=500,
                       help="目标 chunk 字符数（默认 500）")
    p_ing.add_argument("--overlap", type=int, default=75,
                       help="相邻 chunk 重叠字符数（默认 75）")
    p_ing.add_argument("--sentence-aware", action="store_true",
                       help="超长段落优先切在句末/逗号边界，长段落密集的文档建议开启")

    p_ask = sub.add_parser("ask", help="问一个问题")
    p_ask.add_argument("--index", default="index")
    p_ask.add_argument("--q", required=True)
    p_ask.add_argument("--mode", default="hybrid",
                       choices=["hybrid", "vector", "bm25"])
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--show-context", action="store_true")

    p_eval = sub.add_parser("eval", help="跑评测集并输出报告")
    p_eval.add_argument("--index", default="index")
    p_eval.add_argument("--set", default="eval/eval_set.jsonl")
    p_eval.add_argument("--out", default="eval/report.json")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--mode", default="hybrid",
                        choices=["hybrid", "vector", "bm25"])

    p_cmp = sub.add_parser("compare", help="对比 vector / bm25 / hybrid 三种检索")
    p_cmp.add_argument("--index", default="index")
    p_cmp.add_argument("--set", default="eval/eval_set.jsonl")
    p_cmp.add_argument("--out", default="eval/report_compare.json")
    p_cmp.add_argument("--top-k", type=int, default=5)
    p_cmp.add_argument("--retrieval-only", action="store_true",
                       help="只跑检索层指标，不调用生成 API（不消耗额度）")

    args = ap.parse_args()
    rr = None if args.rerank == "off" else args.rerank

    if args.cmd == "ingest":
        use_hash = args.hash_embed if args.hash_embed is not None else USE_HASH_EMBED
        if not use_hash and not API_KEY:
            raise SystemExit(
                "未检测到 LLM_API_KEY。请二选一：\n"
                "  1) 在项目根目录建 .env 并填入 LLM_API_KEY=sk-...\n"
                "  2) 加 --hash-embed 用离线模式（不需要 key，但语义能力弱）"
            )
        ingest(Path(args.docs), Path(args.index), use_hash=use_hash,
               chunk_size=args.chunk_size, overlap=args.overlap,
               sentence_aware=args.sentence_aware)
    elif args.cmd == "ask":
        r = Retriever(Path(args.index))
        if args.show_context:
            for i, (idx, s) in enumerate(
                    r.search(args.q, args.top_k, args.mode, rerank=rr,
                             rerank_pool=args.rerank_pool, alpha=args.alpha), 1):
                c = r.chunks[idx]
                print(f"\n[{i}] score={s:.4f} {c['doc']}/{c['cid']}\n{c['text'][:400]}")
        res = answer(args.q, r, top_k=args.top_k, mode=args.mode, rerank=rr,
                     rerank_pool=args.rerank_pool, alpha=args.alpha)
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif args.cmd == "eval":
        rep = evaluate(Path(args.index), Path(args.set), Path(args.out),
                       top_k=args.top_k, modes=[args.mode],
                       rerank=rr, rerank_pool=args.rerank_pool, alpha=args.alpha)
        print_report(rep)
        print(f"\n报告已写入 {args.out}")
    elif args.cmd == "compare":
        rep = evaluate(Path(args.index), Path(args.set), Path(args.out),
                       top_k=args.top_k, modes=["vector", "bm25", "hybrid"],
                       retrieval_only=args.retrieval_only,
                       rerank=rr, rerank_pool=args.rerank_pool, alpha=args.alpha)
        print_report(rep)
        print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
