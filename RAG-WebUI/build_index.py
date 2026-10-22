# -*- coding: utf-8 -*-
"""按 registry.json 批量建索引。

复用 demo/kb_agent.py 的 ingest()，保证切分与向量化逻辑和 demo 完全一致。

用法：
  python build_index.py                 # 只建还没索引的
  python build_index.py --force         # 全部重建
  python build_index.py --only ai_prep  # 只建指定库
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "RAG知识库Agent" / "demo"
sys.path.insert(0, str(DEMO))
import kb_agent as K  # noqa: E402

REGISTRY = HERE / "kb" / "registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="批量建索引")
    ap.add_argument("--force", action="store_true", help="已存在的也重建")
    ap.add_argument("--only", default=None, help="只建指定 id")
    args = ap.parse_args()

    reg = load_registry()
    print(f"[配置] embedding = {K.EMB_MODEL} | 维度上限取决于模型输出")
    print(f"[配置] base_url  = {K.BASE_URL}\n")

    for kb in reg["kbs"]:
        if args.only and kb["id"] != args.only:
            continue
        docs = HERE / kb["docs"]
        index = HERE / kb["index"]
        meta_f = index / "meta.json"

        if meta_f.exists() and not args.force:
            m = json.loads(meta_f.read_text("utf-8"))
            print(f"[跳过] {kb['id']:12} 已有索引：{m.get('n_chunks')} chunks / "
                  f"{m.get('dim')} 维 / {m.get('built_at')}")
            continue

        print(f"[建库] {kb['id']:12} {kb['name']}")
        t0 = time.time()
        K.ingest(docs, index,
                 chunk_size=kb.get("chunk_size", 500),
                 overlap=kb.get("overlap", 75),
                 sentence_aware=kb.get("sentence_aware", False))
        n = json.loads(meta_f.read_text("utf-8"))["n_chunks"]
        print(f"       耗时 {time.time()-t0:.1f}s\n")

    print("[完成] 索引状态：")
    for kb in reg["kbs"]:
        meta_f = HERE / kb["index"] / "meta.json"
        if meta_f.exists():
            m = json.loads(meta_f.read_text("utf-8"))
            print(f"  {kb['id']:12} {m['n_chunks']:>4} chunks | {m['dim']} 维 | "
                  f"{m['embed_model']} | {m['built_at']}")
        else:
            print(f"  {kb['id']:12} 未建")
    return 0


if __name__ == "__main__":
    sys.exit(main())
