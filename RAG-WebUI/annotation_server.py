# -*- coding: utf-8 -*-
"""人工标注 Web 界面（给 IAA 用）。

为什么单独起一个服务，而不是塞进 server.py
--------------------------------------------
1. server.py 是问答界面，标注是另一件工作，混在一起容易误操作。
2. 标注页面**绝不能**碰到 `kb/*/eval/report_accuracy_*.json`（判据结论）。
   独立服务可以在代码层面保证它只读 `_annotation_sample.json`，读不到别的东西。
3. 两位标注者可以各开一个窗口/浏览器标签，互不影响。

接口
----
  GET  /                    标注页面
  GET  /api/meta            评分标准 + 进度
  GET  /api/sample          全部样本（已剥离判据字段）
  GET  /api/annotations?annotator=名字
  POST /api/save            保存一条或整批标注
  GET  /api/summary         各标注者的进度概况

用法：
  python annotation_server.py                # http://127.0.0.1:8850
  python annotation_server.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
SAMPLE_F = KB / "_annotation_sample.json"
PAGE_F = HERE / "annotate.html"

# 判据字段：绝对不让它们出现在标注页面里
FORBIDDEN = {"grade", "score", "reason", "missing", "missing_list",
             "missing_ungrounded", "covered_list", "judged_answerable",
             "answer_coverage"}

_LOCK = threading.Lock()


def load_sample() -> dict:
    if not SAMPLE_F.exists():
        raise SystemExit(f"找不到样本文件：{SAMPLE_F}\n请先跑 build_annotation_sample.py")
    return json.loads(SAMPLE_F.read_text("utf-8"))


def ann_file(annotator: str) -> Path:
    # 只允许安全字符，防止路径穿越
    safe = "".join(ch for ch in annotator if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("标注者名字无效")
    return KB / f"_annotations_{safe}.json"


def load_annotations(annotator: str) -> dict:
    f = ann_file(annotator)
    if f.exists():
        return json.loads(f.read_text("utf-8")).get("labels", {})
    return {}


def save_annotations(annotator: str, labels: dict) -> None:
    f = ann_file(annotator)
    f.write_text(json.dumps({
        "annotator": annotator,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(labels),
        "labels": labels,
    }, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def sanitize(cases: list[dict]) -> list[dict]:
    """再兜一层：确保返回给页面的样本不含判据字段。"""
    out = []
    for c in cases:
        d = {k: v for k, v in c.items() if k not in FORBIDDEN}
        out.append(d)
    return out


def summary() -> list[dict]:
    sample = load_sample()
    total = len(sample["cases"])
    rows = []
    for f in sorted(KB.glob("_annotations_*.json")):
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:                                   # noqa: BLE001
            continue
        labels = d.get("labels", {})
        rows.append({
            "annotator": d.get("annotator", f.stem),
            "n": len(labels),
            "total": total,
            "saved_at": d.get("saved_at", ""),
        })
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "AnnotationUI/1.0"
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
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
        path, _, query = self.path.partition("?")
        params = {}
        for kv in query.split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k] = v

        try:
            if path in ("/", "/annotate"):
                if not PAGE_F.exists():
                    self._send(500, "annotate.html 缺失", "text/plain; charset=utf-8")
                    return
                self._send(200, PAGE_F.read_text("utf-8"), "text/html; charset=utf-8")

            elif path == "/api/meta":
                s = load_sample()
                self._send(200, {"meta": s["meta"], "n": len(s["cases"]),
                                 "summary": summary()})

            elif path == "/api/sample":
                s = load_sample()
                self._send(200, {"cases": sanitize(s["cases"])})

            elif path == "/api/annotations":
                who = params.get("annotator", "")
                self._send(200, {"annotator": who,
                                 "labels": load_annotations(who) if who else {}})

            elif path == "/api/summary":
                self._send(200, {"summary": summary()})

            else:
                self._send(404, {"error": "not found"})
        except Exception as e:                              # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as e:                              # noqa: BLE001
            self._send(400, {"error": f"解析请求失败: {e}"})
            return

        try:
            if path == "/api/save":
                who = (body.get("annotator") or "").strip()
                if not who:
                    self._send(400, {"error": "缺少 annotator"})
                    return
                labels = body.get("labels") or {}
                if not isinstance(labels, dict):
                    self._send(400, {"error": "labels 必须是对象"})
                    return
                # 只接受合法等级值
                ok = {"CORRECT", "PARTIAL", "INCORRECT"}
                labels = {k: v for k, v in labels.items() if v in ok}
                with _LOCK:
                    save_annotations(who, labels)
                self._send(200, {"ok": True, "n": len(labels)})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:                              # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> int:
    ap = argparse.ArgumentParser(description="人工标注 Web 界面")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8850)
    args = ap.parse_args()

    s = load_sample()
    print(f"[样本] {len(s['cases'])} 条（不含判据字段）")
    existing = summary()
    if existing:
        desc = ", ".join(f"{x['annotator']}({x['n']}/{x['total']})" for x in existing)
    else:
        desc = "无"
    print(f"[已有] {len(existing)} 位标注者：{desc}")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n[就绪] 打开浏览器标注：http://{args.host}:{args.port}")
    print("       两位标注者请各开一个窗口，输入不同名字，互不影响。\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
