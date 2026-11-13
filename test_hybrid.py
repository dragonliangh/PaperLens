"""阶段 2 混合检索验收测试。

【模块职责】
  分两组：
    - 纯逻辑测试（_tokenize / _is_zh / _rrf_fuse）：零依赖，可进 CI。
    - 集成测试（bm25_search / search_hybrid / rebuild_bm25）：依赖 Chroma，本地跑。

【依赖】
  pytest；hybrid_search；index_store（测试里建索引用）。
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

from index_store import Chunk, RetrievedChunk, build_index, clear_index, search
from ingest import parse_and_chunk
from hybrid_search import (
    _is_zh,
    _rrf_fuse,
    _tokenize,
    bm25_search,
    rebuild_bm25,
    search_hybrid,
)

SAMPLE = "data/sample.pdf"
needs_sample = pytest.mark.skipif(not os.path.exists(SAMPLE), reason="缺少 sample.pdf")


# ===========================================================================
# 测试辅助
# ===========================================================================
def _make_chunk(chunk_id: str) -> Chunk:
    """造一个字段合法的 Chunk，仅 chunk_id 有意义（_rrf_fuse 只用它）。"""
    return Chunk(
        text="placeholder",
        doc_name="test.pdf",
        page_start=1,
        page_end=1,
        section="",
        chunk_id=chunk_id,
    )


# ===========================================================================
# 纯逻辑测试（不依赖 Chroma / 模型，可进 CI）
# ===========================================================================
def test_tokenize_basic():
    """_tokenize 提取小写字母数字段，丢弃标点。"""
    assert _tokenize("rq=rv=8") == ["rq", "rv", "8"]
    assert _tokenize("Edward Hu") == ["edward", "hu"]
    assert _tokenize("") == []
    assert _tokenize("!!!") == []


def test_is_zh():
    """_is_zh 含中文即 True。"""
    assert _is_zh("LoRA 减少多少参数") is True
    assert _is_zh("How much does LoRA reduce?") is False
    assert _is_zh("") is False


def test_rrf_fuse_basic():
    """_rrf_fuse：两路都命中时，RRF 分数 = 1/(60+rank_vec) + 1/(60+rank_bm25)。"""
    r1 = RetrievedChunk(chunk=_make_chunk("a"), score=0.9)
    r2 = RetrievedChunk(chunk=_make_chunk("b"), score=0.8)
    bm25 = [_make_chunk("b"), _make_chunk("a")]

    fused = _rrf_fuse([r1, r2], bm25, top_k=2)
    # a: vec rank 1, bm25 rank 2 -> 1/61 + 1/62
    # b: vec rank 2, bm25 rank 1 -> 1/62 + 1/61
    assert abs(fused[0].score - (1/61 + 1/62)) < 1e-9
    assert abs(fused[1].score - (1/61 + 1/62)) < 1e-9


def test_rrf_fuse_one_side_only():
    """只在某一路出现的 chunk，另一路不贡献分数。"""
    r1 = RetrievedChunk(chunk=_make_chunk("a"), score=0.9)
    r2 = RetrievedChunk(chunk=_make_chunk("b"), score=0.8)
    bm25 = [_make_chunk("c")]

    fused = _rrf_fuse([r1, r2], bm25, top_k=3)
    by_id = {r.chunk.chunk_id: r.score for r in fused}
    assert abs(by_id["a"] - 1/61) < 1e-9          # 只有向量路
    assert abs(by_id["c"] - 1/61) < 1e-9          # 只有 BM25 路
    assert fused[0].score >= fused[-1].score


# ===========================================================================
# 集成测试（依赖 Chroma）
# ===========================================================================
@pytest.fixture(scope="module")
def indexed():
    """建索引 + rebuild BM25，供本模块集成测试共享。"""
    clear_index()
    chunks = parse_and_chunk(SAMPLE)
    build_index(chunks)
    rebuild_bm25()
    return chunks


@needs_sample
def test_bm25_exact_string(indexed):
    """契约 18a：BM25 对正文精确词（AdamW）能命中含该 token 的 chunk。

    【用例修正 C34】原用例 rq=rv=8 失败：诊断发现 84 个 chunk 里 rq/rv token 数为 0，
    根因是 PDF 对下标 q/v 的字符提取不可靠，rq=rv=8 未被完整抽出（第 1 块已知限制）。
    改用正文词 AdamW（第 19 页 GPT-3 实验配置，PDF 能可靠抽出）。
    BM25 的匹配单位是 token：query "AdamW" token 化为 ["adamw"]，top-1 应含该 token。
    """
    from hybrid_search import _tokenize
    results = bm25_search("AdamW", top_k=5)
    assert len(results) > 0
    top_tokens = set(_tokenize(results[0].text))
    assert "adamw" in top_tokens, (
        f"top-1 chunk 未含 adamw token，实际 tokens 示例：{sorted(top_tokens)[:20]}"
    )

@needs_sample
def test_bm25_chinese_returns_empty(indexed):
    """契约 18b：中文 query 走 bm25_search 返回 []。"""
    assert bm25_search("LoRA 减少多少参数") == []


@needs_sample
def test_bm25_empty_query(indexed):
    """契约 8a：空 query 返回 []。"""
    assert bm25_search("") == []
    assert bm25_search("   ") == []


@needs_sample
def test_search_hybrid_english(indexed):
    """契约 18c：英文 query 返回 ≤ top_k 条，按 RRF 分数降序。"""
    results = search_hybrid("How much does LoRA reduce trainable parameters?", top_k=5)
    assert 0 < len(results) <= 5
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@needs_sample
def test_search_hybrid_chinese_passthrough(indexed):
    """契约 18d：中文 query 与 search() 完全一致（透传验证）。"""
    q = "LoRA 减少多少参数"
    hybrid = search_hybrid(q, top_k=5)
    vec = search(q, top_k=5)
    assert len(hybrid) == len(vec)
    for h, v in zip(hybrid, vec):
        assert h.chunk.chunk_id == v.chunk.chunk_id
        assert abs(h.score - v.score) < 1e-9


@needs_sample
def test_search_hybrid_empty(indexed):
    """契约 8a：空 query 返回 []。"""
    assert search_hybrid("") == []
    assert search_hybrid("   ") == []


@needs_sample
def test_rebuild_sync(indexed):
    """契约 18f：rebuild 后 bm25_search 能检索到（正向同步验证）。

    【测试局限】"不调 rebuild 就检索不到"较难在单测试内干净构造
    （懒加载会自动读到 Chroma）。该脆弱点靠契约 + README 约束，不强行构造反例。
    """
    clear_index()
    rebuild_bm25()
    assert bm25_search("LoRA") == []

    chunks = parse_and_chunk(SAMPLE)
    build_index(chunks)
    rebuild_bm25()
    assert len(bm25_search("LoRA", top_k=5)) > 0

