"""第 2 块验收测试：对应契约 11a-11g。

运行方式（项目根目录、已激活 .venv）：
    pytest test_index_store.py -v

注意：本测试会真实加载嵌入模型、写入 Chroma，首次运行较慢（几十秒）。
测试结束后索引保留，方便人工查看；跑 11f 会清空索引。
"""
import os
# 强制离线，必须在 import transformers 相关模块之前设置。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

from ingest import parse_and_chunk
from index_store import (
    _get_collection,   # 测试里用私有函数读 count，仅测试用
    build_index,
    clear_index,
    search,
)

SAMPLE = "data/sample.pdf"
needs_sample = pytest.mark.skipif(
    not os.path.exists(SAMPLE), reason="缺少 data/sample.pdf"
)


@pytest.fixture(scope="module")
def sample_chunks():
    """解析一次 sample.pdf，供多个测试共享，避免重复解析。"""
    return parse_and_chunk(SAMPLE)


@pytest.fixture(scope="module")
def indexed(sample_chunks):
    """清空索引后重建一次，供依赖索引的测试共享。

    module scope：整个测试模块只建一次索引，避免每个测试都嵌入 84 个 chunk。
    """
    clear_index()
    build_index(sample_chunks)
    return sample_chunks


@needs_sample
def test_11a_index_size(indexed):
    """契约 11a：建索引不报错，Collection 条数等于 chunk 数。"""
    assert _get_collection().count() == len(indexed)


@needs_sample
def test_11b_search_top3(indexed):
    """契约 11b：相关问题检索 top_k=3，返回 3 条，score 递减。"""
    results = search("How much does LoRA reduce trainable parameters?", top_k=3)
    assert len(results) == 3
    assert results[0].score >= results[1].score >= results[2].score


@needs_sample
def test_11c_first_hit_page(indexed):
    """契约 11c：第 1 条结果的 page_start 落在第 1-2 页。

    LoRA 摘要和引言（第 1-2 页）明确写了「reduce by 10,000 times」「0.01%」，
    所以「减少多少参数」这个问题的高分命中应该在前两页。
    """
    results = search("How much does LoRA reduce trainable parameters?", top_k=3)
    assert results[0].chunk.page_start <= 2


@needs_sample
def test_11d_unrelated_lower(indexed):
    """契约 11d：无关问题的最高分明显低于相关问题的最高分。

    不写死阈值（阈值要靠第 7 块实测），只断言相对关系。
    """
    related = search("How much does LoRA reduce trainable parameters?", top_k=1)
    unrelated = search("What is the capital of France?", top_k=1)
    assert unrelated[0].score < related[0].score


@needs_sample
def test_11e_rebuild_idempotent(indexed):
    """契约 11e：同一篇论文重复 build_index 两次，条数不变（幂等）。"""
    before = _get_collection().count()
    build_index(indexed)  # 再建一次
    after = _get_collection().count()
    assert after == before == len(indexed)


@needs_sample
def test_11g_empty_query(indexed):
    """契约 11g：空 query 返回 []，即使索引里有数据。"""
    assert search("") == []
    assert search("   ") == []  # 纯空白也算空


@needs_sample
def test_11f_empty_index(sample_chunks):
    """契约 11f：空索引检索返回 []。

    这个测试放最后：它会清空索引，避免影响前面依赖索引的测试。
    """
    clear_index()
    assert _get_collection().count() == 0
    assert search("How much does LoRA reduce trainable parameters?") == []