"""PaperLens 阶段 2：混合检索（BM25 + 向量 + RRF）。

【模块职责】
  在现有向量检索之外，新增 BM25 关键词检索一路，用 RRF 融合两路排名，
  补向量检索对精确字符串（表格配置、符号、缩写）不敏感的短板。
  中文 Query 只走向量（不做 BM25），理由见契约 2.7。

【在架构中的位置】
  检索层增强，位于 index_store（第 2 块）与 qa（第 3/6/7 块）之间。
    - 上游依赖：index_store.search（向量路）、index_store._get_collection（读 Chroma 建 BM25）。
    - 下游被依赖：qa.answer（通过 `import as search` 别名接入，签名不变）。
    - 明确不修改 index_store.py / rerank.py / evaluate.py / ingest.py。

【类级】
  本文件无类定义，全部为模块级函数 + 一个模块级全局状态 _BM25_STATE。

【对外暴露】
  - bm25_search(query, top_k=20) -> list[Chunk]
  - search_hybrid(query, top_k=20) -> list[RetrievedChunk]
  - rebuild_bm25() -> None

【遵循的契约（阶段 2）】
  - 2.2.1 bm25_search：英文 BM25 检索，中文返回 []，空输入/空索引返回 []。
  - 2.2.2 search_hybrid：英文 RRF 融合，中文透传 search()；score 语义见 2.2.2。
  - 2.2.3 rebuild_bm25：从 Chroma 读全部 chunk 重建内存 BM25 索引。
  - 2.3 RRF：k=60，按 chunk_id 去重，按 RRF 分数降序。
  - 2.4 生命周期：从 Chroma 读 + 首次懒加载 + 显式 rebuild。
  - 2.5 分词：英文正则 [a-z0-9]+；不引入 jieba。
  - 2.7 中文不做 BM25，写进 README。
  - 2.8 边界 8a-8e；2.9 错误不吞。

【遵循的设计决策】
  - H1：BM25 状态用模块级全局变量，不用 lru_cache（重建语义直白）。
  - H2：_is_zh 内联正则，不 import qa（避免循环依赖）。
  - H3：_rrf_fuse 用 dict 排名映射，缺失路不贡献分数。
  - H4：bm25_search 返回 list[Chunk]，不带分数。
  - H5：search_hybrid 对 zh 直接透传 search()。
  - H8：空 corpus 时 bm25=None，bm25_search 返回 []。
  - 契约缺口 a：从 index_store import _get_collection（同项目内共享 client）。

【依赖】
  标准库 re；rank_bm25（BM25Okapi）；index_store（search / _get_collection / RetrievedChunk / Chunk）。
"""
import re

from rank_bm25 import BM25Okapi

from index_store import (
    Chunk,
    RetrievedChunk,
    _get_collection,   # 契约缺口 a：复用 index_store 的 client 单例，避免两个 PersistentClient 锁冲突
    search as _vector_search,
)

# BM25 索引内存态：None 表示尚未建（或 Chroma 为空）。
# 【H1】用模块级全局变量而非 lru_cache：rebuild_bm25 需要"强制扔掉重建"，
#       显式赋值语义直白；lru_cache 的 cache_clear + 重调语义绕且有竞态风险。
_BM25_STATE: dict | None = None

# RRF 公式常数，论文默认值。
_RRF_K = 60

# 中文字符正则：与 qa._detect_lang 口径一致，但内联一份避免循环 import（H2）。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 英文 token 正则：只取小写字母数字连续段。
# 例："rq=rv=8" -> ["rq", "rv", "8"]；"Edward Hu" -> ["edward", "hu"]。
# 【边界】下划线、连字符、Unicode 字母（如 é）会被切断，当前英文论文场景可接受。
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """把文本切成 BM25 用的 token 列表（小写字母数字段）。

    参数：
      text : 任意文本（英文 chunk 原文或英文 query）。

    返回：
      token 列表。空文本返回 []。
    """
    return _TOKEN_RE.findall(text.lower())


def _is_zh(query: str) -> bool:
    """判断 query 是否含中文；含则视为中文 query（走纯向量路）。"""
    return _CJK_RE.search(query) is not None


def _build_from_chroma() -> dict | None:
    """从 Chroma 读全部 chunk，构建 BM25 索引内存态。

    返回：
      {"bm25": BM25Okapi | None, "chunks": list[Chunk]}；
      Chroma 为空时返回 None（契约 8b：空索引不报错）。

    错误语义：
      Chroma 读取失败抛异常，不吞（契约 2.9）。
    """
    col = _get_collection()
    # 【性能隐患】全量 get 一次拉全部 documents+metadatas。
    # 当前 84 chunk 毫秒级；若未来扩到万级需改为分页。
    data = col.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    if not docs:
        return None  # 契约 8b：空 Chroma，bm25_search 后续返回 []

    # 【核心逻辑】按 Chroma 返回顺序重建 Chunk 列表，与 tokenized_corpus 一一对应。
    chunks = [
        Chunk(
            text=doc,
            doc_name=meta["doc_name"],
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            section=meta["section"],
            chunk_id=meta["chunk_id"],
        )
        for doc, meta in zip(docs, metas)
    ]
    tokenized = [_tokenize(c.text) for c in chunks]

    # 【边界】若全部 chunk 都 tokenize 成空（极端异常输入），BM25Okapi 会因空 corpus 报错，
    # 此处降级为 None，与"空 Chroma"同语义：bm25_search 返回 []。
    if not any(tokenized):
        return None

    return {"bm25": BM25Okapi(tokenized), "chunks": chunks}


def rebuild_bm25() -> None:
    """强制重建 BM25 索引（从 Chroma 读全量 chunk）。

    调用约定（契约 2.4 / 防法 1）：
      build_index() 之后必须调用本函数，否则 BM25 索引与 Chroma 不同步。
      唯一生产入口是 app.py 的 _on_upload，两行紧挨着写。
    """
    global _BM25_STATE
    _BM25_STATE = _build_from_chroma()


def _get_state() -> dict | None:
    """懒加载 BM25 状态（H1：模块级变量，不用 lru_cache）。"""
    global _BM25_STATE
    if _BM25_STATE is None:
        # 首次调用触发建表；若 Chroma 也为空，则保持 None，后续调用直接返回 []。
        _BM25_STATE = _build_from_chroma()
    return _BM25_STATE


def bm25_search(query: str, top_k: int = 20) -> list[Chunk]:
    """BM25 关键词检索（仅英文 query）。

    参数：
      query : 用户问题；含中文时直接返回 []（契约 2.2.1 / 2.7）。
      top_k : 返回条数上限，≥1。

    返回：
      list[Chunk]，按 BM25 分数降序，长度 ≤ top_k，不带分数（H4）。

    边界：
      - 空 query → []（契约 8a）。
      - 中文 query → []（契约 8c）。
      - 空 Chroma → []（契约 8b）。
      - top_k 超过候选数 → 返回全部候选（契约 8e）。
    """
    # 【核心逻辑】中文不做 BM25（契约 2.7）：直接短路，避免噪声投票。
    if not query.strip() or _is_zh(query):
        return []

    state = _get_state()
    if state is None:
        return []  # 契约 8b：空索引

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        # 边界：query 里没有任何字母数字（如纯标点），BM25 无法打分，返回 []。
        return []

    scores = state["bm25"].get_scores(tokenized_query)
    chunks = state["chunks"]

    # 【核心逻辑】按 BM25 分数降序取 top_k。
    # 用 sorted + 切片而非 argpartition：84 条候选规模下开销可忽略，代码直白。
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked_idx[:top_k]]


def _rrf_fuse(
    vec_results: list[RetrievedChunk],
    bm25_chunks: list[Chunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """RRF 融合两路排名，返回按 RRF 分数降序的 RetrievedChunk 列表。

    公式：rrf_score(chunk) = Σ 1/(k + rank_i)，k=60，rank_i 为 1-based 排名。
    只在某一路出现的 chunk，该路不贡献分数（H3：用 dict 查缺失，O(1)）。

    参数：
      vec_results  : 向量路结果（带 score，这里只用排名）。
      bm25_chunks  : BM25 路结果（不带分数，顺序即排名）。
      top_k        : 返回条数上限。

    返回：
      list[RetrievedChunk]，score 字段为 RRF 分数（约 0~0.033，非余弦）。
      **契约 2.2.2：此 score 不得直接用于拒答阈值判断，下游 rerank 必须覆盖。**
    """
    # 两路各自的 1-based 排名映射：chunk_id -> rank
    vec_rank = {r.chunk.chunk_id: i for i, r in enumerate(vec_results, start=1)}
    bm25_rank = {c.chunk_id: i for i, c in enumerate(bm25_chunks, start=1)}

    # chunk_id -> Chunk：优先取向量路的（两路同一 chunk 内容一致，取任一路即可）
    chunk_by_id: dict[str, Chunk] = {r.chunk.chunk_id: r.chunk for r in vec_results}
    for c in bm25_chunks:
        chunk_by_id.setdefault(c.chunk_id, c)

    all_ids = set(vec_rank) | set(bm25_rank)

    # 【核心逻辑】RRF 分数累加：每路命中则加 1/(k+rank)，未命中不加。
    scores: dict[str, float] = {}
    for cid in all_ids:
        s = 0.0
        if cid in vec_rank:
            s += 1.0 / (_RRF_K + vec_rank[cid])
        if cid in bm25_rank:
            s += 1.0 / (_RRF_K + bm25_rank[cid])
        scores[cid] = s

    # 按 RRF 分数降序，截 top_k
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [
        RetrievedChunk(chunk=chunk_by_id[cid], score=scores[cid])
        for cid in ranked_ids[:top_k]
    ]


def search_hybrid(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    """混合检索主入口：英文走 RRF 融合，中文透传向量检索。

    参数：
      query : 用户问题。
      top_k : 返回条数上限。

    返回：
      list[RetrievedChunk]，长度 ≤ top_k，按 RRF 分数（英文）或余弦分数（中文）降序。

    score 语义（契约 2.2.2）：
      - 英文：RRF 分数，范围约 0~0.033，**不是余弦相似度**；下游 rerank 必须覆盖。
      - 中文：向量余弦分数，与现状一致。
      - **降级路径不得用 score 做阈值判断**（阶段 4 契约 C3）。

    边界：
      - 空 query → []（契约 8a）。
      - 空索引 → []（契约 8b）。
    """
    if not query.strip():
        return []  # 契约 8a

    # 【H5 / 契约 2.7】中文只走向量，透传，保证与现状 100% 一致。
    if _is_zh(query):
        return _vector_search(query, top_k=top_k)

    # 英文：两路检索 + RRF 融合。
    vec_results = _vector_search(query, top_k=top_k)
    bm25_chunks = bm25_search(query, top_k=top_k)

    # 边界：两路都空（空索引）→ 返回 []
    if not vec_results and not bm25_chunks:
        return []

    return _rrf_fuse(vec_results, bm25_chunks, top_k)