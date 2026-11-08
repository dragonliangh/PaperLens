"""PaperLens 第 2 块：嵌入 + 向量库索引。

职责：
  - build_index(chunks)   ：把第 1 块的 Chunk 列表嵌入成向量，写入 Chroma。
  - search(query, top_k)  ：把问题嵌入，从 Chroma 检索最相似的 K 个 chunk。
  - clear_index()         ：清空整个索引。

设计约定（详见 DESIGN.md 第 2 块）：
  - 嵌入模型：sentence-transformers 加载本地 BAAI/bge-small-en-v1.5。
  - 向量库：Chroma PersistentClient，数据存 data/chroma/。
  - 共用一个 Collection "papers"，用 metadata 的 doc_name 区分文档。
  - 向量 L2 归一化，距离度量 cosine，score = 1 - distance。
  - 幂等：同一文档重复建索引，先删旧数据再写入。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from ingest import Chunk
from typing import Any

# 模型本地路径（与第 1 块 tokenizer 同一个目录）。
# ST 能直接从这个目录加载，因为 ModelScope 下载的是完整的 ST 格式模型。
MODEL_PATH = "./models/bge-small-en-v1.5"
# Chroma 数据目录，相对项目根目录。
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "papers"
# 嵌入批大小：契约要求分批，避免一次性大量文本爆内存。
EMBED_BATCH_SIZE = 32


@dataclass
class RetrievedChunk:
    """检索结果：一个 chunk + 它与 query 的相似度分数。

    score = 1 - cosine_distance，范围约 [-1, 1]，越大越相似。
    注意：score 是相对排序依据，不是「概率」，不要直接当置信度用。
    """
    chunk: Chunk
    score: float


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """加载并缓存嵌入模型，进程内只加载一次。

    性能隐患：
      首次加载约 130MB 模型，需要几秒到十几秒；之后走 lru_cache 复用。
      如果 MODEL_PATH 下文件缺失，这里会抛异常（契约：不吞）。

    【云端修复】
      由于云端在启动时已通过 modelscope 下载了完整的 .safetensors 文件，
      这里直接使用默认的 safetensors 加载，既快又安全，彻底绕开 PyTorch 2.6 
      的 weights_only 限制。
    """
    return SentenceTransformer(MODEL_PATH)


@lru_cache(maxsize=1)
def _get_client() -> Any:
    # 【已知缺陷】chromadb 的类型存根里 PersistentClient 是一个工厂函数，
    # 但 Pylance 期待一个类，会导致报错。这里用 Any 规避静态检查。
    """Chroma 客户端单例，数据持久化到 CHROMA_PATH。

    为什么用单例：
      同一路径重复打开 PersistentClient 可能触发 SQLite 锁冲突，
      lru_cache 保证一个进程内只 open 一次。
    """
    # 确保目录存在；Chroma 通常会自动建，显式建一次更稳。
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection():
    """获取（或创建）Collection。

    不缓存：clear_index 会删掉 Collection 再重建，
    缓存对象会失效，所以每次调用都 get_or_create。
    """
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 距离度量用余弦，配合 L2 归一化
    )


def _embed(texts: list[str]) -> list[list[float]]:
    """把一批文本编码成 L2 归一化的向量列表。

    【核心逻辑】
      - encode 内部按 batch_size 分批，避免一次性爆内存；
      - normalize_embeddings=True 做 L2 归一化，使余弦相似度等价于点积；
      - .tolist() 把 numpy array 转成 Chroma 接受的 list of list of float。
    """
    vectors = _get_model().encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def build_index(chunks: list[Chunk]) -> None:
    """把 chunks 嵌入并写入 Chroma。同一文档重复调用是幂等的。

    契约：
      - 空列表：直接返回，不报错（建空索引）。
      - 幂等：先按 doc_name 删除该文档旧数据，再写入新数据。
    """
    if not chunks:
        return  # 契约 2：空列表建空索引，不报错

    # 【假设特定输入格式】同一批 chunks 属于同一篇论文，
    # 用第一条的 doc_name 作为删除依据。若传入混合文档的 chunks，
    # 这条幂等逻辑不成立（只会删掉第一个文档的旧数据）。
    doc_name = chunks[0].doc_name
    collection = _get_collection()
    # 幂等：先删旧数据。Chroma 对无匹配的 delete 是安静的，不会报错。
    collection.delete(where={"doc_name": doc_name})

    texts = [c.text for c in chunks]
    embeddings = _embed(texts)  # 契约：分批 32，L2 归一化

    collection.add(
        ids=[c.chunk_id for c in chunks],          # id 用 chunk_id（全局唯一）
        embeddings=embeddings,
        documents=texts,                            # 原文存 document 字段
        metadatas=[
            {
                "doc_name": c.doc_name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ],
    )


def search(query: str, top_k: int = 8) -> list[RetrievedChunk]:
    """检索与 query 最相似的 top_k 个 chunk，按相似度降序返回。

    契约：
      - 空 query：返回 []。
      - 空索引：返回 []。
      - 返回长度 ≤ top_k，按 score 降序。
    """
    if not query.strip():
        return []  # 契约 8d：空 query 返回 []
    collection = _get_collection()
    if collection.count() == 0:
        return []  # 契约 8a：空索引返回 []

    query_vec = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma 返回的是嵌套列表（外层对应每个 query），我们只有一个 query，取 [0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunk = Chunk(
            text=doc,
            doc_name=meta["doc_name"],
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            section=meta["section"],
            chunk_id=meta["chunk_id"],
        )
        # 【核心逻辑】Chroma 返回 cosine distance（越小越近），
        # 转成相似度（越大越近），符合直觉，也方便第 7 块设拒答阈值。
        score = 1.0 - dist
        out.append(RetrievedChunk(chunk=chunk, score=score))
    return out


def clear_index() -> None:
    """清空整个索引：删除 Collection 后重建。

    为什么删掉重建而不是逐条删：
      空 where 的 delete 在不同 Chroma 版本行为不一致；
      删 Collection 再 get_or_create 最干净，也保证名字不变。
    """
    client = _get_client()
    # list_collections() 在新版 Chroma 返回 Collection 对象列表，取 .name。
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    _get_collection()  # 重建空 collection，保证后续调用拿到的还是同一个名字