# -*- coding: utf-8 -*-
"""RAG 知识库 —— 本地 Web 前端服务（零第三方依赖）

复用 demo/kb_agent.py 的检索与生成能力，只加一层 HTTP 外壳：
  GET  /                -> 前端页面
  GET  /api/config      -> 配置 + 知识库清单
  POST /api/ask         -> 提问：检索 + 生成（一次性返回）
  POST /api/ask_stream  -> 提问：流式（NDJSON，逐字返回）
  POST /api/search      -> 只检索，不调生成（免费、快）

请求体都可带 "kb" 字段选择知识库，取值见 kb/registry.json。

用法：
  python server.py                 # 默认 http://127.0.0.1:8848
  python server.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "RAG知识库Agent" / "demo"

if not (DEMO / "kb_agent.py").exists():
    raise SystemExit(f"找不到 kb_agent.py，请确认路径：{DEMO}")

sys.path.insert(0, str(DEMO))
import kb_agent as K  # noqa: E402

# ---------------------------------------------------------------------------
# 多知识库注册表 + 按需加载
#
# 每个库的 Retriever 只在第一次被用到时构建（BM25 词表 + 载入向量），
# 之后缓存复用。这样启动快，也不会为了没用到的库白读一遍数据。
# ---------------------------------------------------------------------------
REGISTRY_FILE = HERE / "kb" / "registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY_FILE.read_text("utf-8"))


REGISTRY = load_registry()
KB_BY_ID = {k["id"]: k for k in REGISTRY["kbs"]}
DEFAULT_KB = REGISTRY.get("default") or REGISTRY["kbs"][0]["id"]

_RETRIEVERS: dict[str, K.Retriever] = {}
_LOCK = threading.Lock()

print(f"[配置] base_url = {K.BASE_URL}")
print(f"[配置] llm_model = {K.LLM_MODEL} | emb_model = {K.EMB_MODEL}")
print(f"[配置] api_key   = {'已配置 (' + K.API_KEY[:6] + '...)' if K.API_KEY else '缺失'}")
print(f"[配置] 知识库    = {len(KB_BY_ID)} 个，默认 {DEFAULT_KB}")


def get_retriever(kb_id: str | None) -> tuple[str, K.Retriever]:
    """取指定知识库的检索器；kb_id 为空或非法时回退到默认库。"""
    if not kb_id or kb_id not in KB_BY_ID:
        kb_id = DEFAULT_KB
    if kb_id not in _RETRIEVERS:
        index_dir = HERE / KB_BY_ID[kb_id]["index"]
        if not (index_dir / "meta.json").exists():
            raise RuntimeError(f"知识库 {kb_id} 还没建索引，请先跑 build_index.py")
        t0 = time.time()
        _RETRIEVERS[kb_id] = K.Retriever(index_dir)
        r = _RETRIEVERS[kb_id]
        print(f"[索引] 加载 {kb_id}: {r.meta['n_chunks']} chunks / "
              f"{r.meta['dim']} 维（{time.time()-t0:.2f}s）")
    return kb_id, _RETRIEVERS[kb_id]


def kb_list() -> list[dict]:
    """给前端的知识库清单，附带是否已建索引。"""
    out = []
    for k in REGISTRY["kbs"]:
        meta_f = HERE / k["index"] / "meta.json"
        item = {kk: k[kk] for kk in ("id", "name", "desc", "emoji", "examples")
                if kk in k}
        item["ready"] = meta_f.exists()
        if item["ready"]:
            m = json.loads(meta_f.read_text("utf-8"))
            item["n_chunks"] = m["n_chunks"]
            item["dim"] = m["dim"]
            item["built_at"] = m.get("built_at")
        out.append(item)
    return out


def do_search(retriever: K.Retriever, query: str, top_k: int, mode: str,
              rerank: str | None, rerank_pool: int, alpha: float) -> list[dict]:
    """检索并返回带原文的命中明细。"""
    hits = retriever.search(query, top_k=top_k, mode=mode, rerank=rerank,
                            rerank_pool=rerank_pool, alpha=alpha)
    out = []
    for rank, (idx, score) in enumerate(hits, start=1):
        c = retriever.chunks[idx]
        out.append({
            "rank": rank,
            "score": round(float(score), 4),
            "cid": c["cid"],
            "doc": c["doc"],
            "text": c["text"],
        })
    return out


def handle_ask(body: dict) -> dict:
    query = (body.get("q") or "").strip()
    if not query:
        return {"error": "问题不能为空"}

    top_k = int(body.get("top_k", 3))
    mode = body.get("mode", "hybrid")
    rerank = body.get("rerank") or None
    if rerank == "off":
        rerank = None
    rerank_pool = int(body.get("rerank_pool", 40))
    alpha = float(body.get("alpha", 0.7))

    t0 = time.time()
    with _LOCK:                      # kb_agent 的搜索内部有共享状态，串行更稳
        kb_id, retriever = get_retriever(body.get("kb"))
        hits = do_search(retriever, query, top_k, mode, rerank, rerank_pool, alpha)
        t_retrieval = time.time() - t0
        res = K.answer(query, retriever, top_k=top_k, mode=mode, rerank=rerank,
                       rerank_pool=rerank_pool, alpha=alpha)
    t_total = time.time() - t0

    return {
        "kb": kb_id,
        "q": query,
        "answerable": res.get("answerable"),
        "answer": res.get("answer"),
        "citations": res.get("citations", []),
        "missing": res.get("missing", ""),
        "repairs": res.get("_repairs", 0),
        "failed": res.get("_failed", False),
        "raw": res.get("_raw", ""),
        "hits": hits,
        "timing": {"retrieval_s": round(t_retrieval, 2),
                   "total_s": round(t_total, 2)},
    }


def stream_ask(body: dict, wfile):
    """NDJSON 流式问答：每行一个 JSON 事件。

    事件类型：
      hits       检索完成（含召回明细）
      reasoning  思维链增量（仅当模型未被禁用思考时出现）
      delta      答案正文增量
      reset      上一次输出校验失败，前端应清空重来
      result     最终结构化结果
      error      出错
    """
    def emit(obj: dict) -> None:
        wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        wfile.flush()

    query = (body.get("q") or "").strip()
    if not query:
        emit({"type": "error", "error": "问题不能为空"})
        return

    top_k = int(body.get("top_k", 3))
    mode = body.get("mode", "hybrid")
    rerank = body.get("rerank") or None
    if rerank == "off":
        rerank = None
    rerank_pool = int(body.get("rerank_pool", 40))
    alpha = float(body.get("alpha", 0.7))

    try:
        t0 = time.time()
        with _LOCK:
            kb_id, retriever = get_retriever(body.get("kb"))
            hits = do_search(retriever, query, top_k, mode, rerank, rerank_pool, alpha)
            t_retrieval = time.time() - t0
            emit({"type": "hits", "kb": kb_id, "hits": hits,
                  "timing": {"retrieval_s": round(t_retrieval, 2)}})

            result = None
            for kind, payload in K.ask_stream(query, retriever, top_k=top_k, mode=mode,
                                              rerank=rerank, rerank_pool=rerank_pool,
                                              alpha=alpha):
                if kind == "delta":
                    emit({"type": "delta", "text": payload})
                elif kind == "reasoning":
                    emit({"type": "reasoning", "text": payload})
                elif kind == "reset":
                    emit({"type": "reset", "error": payload})
                elif kind == "result":
                    result = payload

        emit({"type": "result",
              "kb": kb_id,
              "answerable": result.get("answerable"),
              "answer": result.get("answer"),
              "citations": result.get("citations", []),
              "missing": result.get("missing", ""),
              "repairs": result.get("_repairs", 0),
              "failed": result.get("_failed", False),
              "raw": result.get("_raw", ""),
              "timing": {"retrieval_s": round(t_retrieval, 2),
                         "total_s": round(time.time() - t0, 2)}})
    except Exception as e:                                  # noqa: BLE001
        traceback.print_exc()
        try:
            emit({"type": "error", "error": f"{type(e).__name__}: {e}"})
        except Exception:                                   # noqa: BLE001
            pass


def handle_search(body: dict) -> dict:
    query = (body.get("q") or "").strip()
    if not query:
        return {"error": "问题不能为空"}
    t0 = time.time()
    with _LOCK:
        kb_id, retriever = get_retriever(body.get("kb"))
        hits = do_search(retriever, query, int(body.get("top_k", 5)),
                         body.get("mode", "hybrid"),
                         None if body.get("rerank") in ("off", None) else body.get("rerank"),
                         int(body.get("rerank_pool", 40)), float(body.get("alpha", 0.7)))
    return {"kb": kb_id, "q": query, "hits": hits,
            "timing": {"retrieval_s": round(time.time() - t0, 2)}}


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGWebUI/1.0"
    # 用 HTTP/1.0：流式响应靠「连接关闭」作为结束标志，不需要自己做 chunked 分帧。
    # （BaseHTTPRequestHandler 不会自动做 chunked 编码，
    #   声明了 Transfer-Encoding: chunked 却不分帧，客户端会一直等下去。）
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):        # 让日志干净一点
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]}\n")

    def _send(self, code: int, payload, ctype="application/json; charset=utf-8"):
        if isinstance(payload, (dict, list)):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = payload
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = HERE / "index.html"
            if not page.exists():
                self._send(500, "index.html 缺失", "text/plain; charset=utf-8")
                return
            self._send(200, page.read_text("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._send(200, {
                "llm_model": K.LLM_MODEL,
                "emb_model": K.EMB_MODEL,
                "base_url": K.BASE_URL,
                "has_key": bool(K.API_KEY),
                "disable_thinking": K.DISABLE_THINKING,
                "default_kb": DEFAULT_KB,
                "kbs": kb_list(),
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            body = json.loads(raw or "{}")
        except Exception as e:                                  # noqa: BLE001
            self._send(400, {"error": f"请求体解析失败: {e}"})
            return

        try:
            if path == "/api/ask":
                self._send(200, handle_ask(body))
            elif path == "/api/ask_stream":
                # 流式：HTTP/1.0 下不发 Content-Length，连接关闭即表示流结束
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                stream_ask(body, self.wfile)
            elif path == "/api/search":
                self._send(200, handle_search(body))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:                                  # noqa: BLE001
            traceback.print_exc()
            try:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:                                   # noqa: BLE001
                pass


def main():
    ap = argparse.ArgumentParser(description="RAG 知识库本地 Web 前端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8848)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n[就绪] 打开浏览器访问： {url}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 已停止")


if __name__ == "__main__":
    main()
