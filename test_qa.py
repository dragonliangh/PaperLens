"""第 3 块验收测试：对应契约 9a-9f。

【运行前置条件】
  1. 已执行 pip install zhipuai；
  2. 当前终端已设置 $env:ZHIPU_API_KEY；
  3. data/sample.pdf 存在。

【运行方式】
  pytest test_qa.py -v

【注意事项】
  - 9a/9b 会真实调用 GLM API，耗时几秒到十几秒；
  - 9c/9d 不调 LLM，很快；
  - 9e 会临时删除环境变量，测试后自动恢复；
  - 9f 是人工检查，不写自动断言。
"""
import os
import time

import pytest

from ingest import parse_and_chunk
from index_store import build_index, clear_index
from qa import REFUSAL_TEXT, _get_client, answer

SAMPLE = "data/sample.pdf"
QUESTION = "How much does LoRA reduce trainable parameters?"

needs_sample = pytest.mark.skipif(
    not os.path.exists(SAMPLE), reason="缺少 data/sample.pdf"
)
needs_key = pytest.mark.skipif(
    not os.environ.get("ZHIPU_API_KEY"), reason="缺少 ZHIPU_API_KEY 环境变量"
)


@pytest.fixture(scope="module")
def indexed():
    """清空并重建索引，供本模块测试共享。

    module scope：整个模块只建一次，避免每个测试都嵌入 84 个 chunk。
    """
    chunks = parse_and_chunk(SAMPLE)
    clear_index()
    build_index(chunks)
    return chunks


@pytest.fixture(scope="module")
def rag_result(indexed):
    """真实调用一次 answer()，供 9a/9b 共享，避免重复消耗 API 额度。

    调用前 sleep 2s：契约约定评估类调用之间要留间隔，降低触发限流的概率。
    """
    time.sleep(2)
    return answer(QUESTION)


@needs_sample
@needs_key
def test_9a_answer_and_citations(rag_result):
    """契约 9a：正常问题返回非空 answer 和非空 citations。"""
    assert rag_result.answer.strip()
    assert len(rag_result.citations) > 0


@needs_sample
@needs_key
def test_9b_answer_contains_key_number(rag_result):
    """契约 9b：答案中至少提到一个关键数字。

    放宽了多种写法（带逗号/不带逗号/百分比/大小），
    避免因 LLM 输出格式差异导致误判。
    """
    ans = rag_result.answer
    keywords = ["10,000", "10000", "0.01%", "350GB", "35MB"]
    assert any(kw in ans for kw in keywords), (
        f"答案中未找到任何关键数字，实际答案：{ans}"
    )


@needs_sample
def test_9c_empty_query(indexed):
    """契约 9c：空 query 返回拒答文本，且不调 LLM。

    不调 LLM 的证据：这个测试没有 needs_key 标记，
    在没设 Key 的环境下也应当通过（因为 search 返回 []，直接走拒答分支）。
    """
    r = answer("")
    assert r.answer == REFUSAL_TEXT
    assert r.citations == []


@needs_sample
def test_9d_empty_index(indexed):
    """契约 9d：清空索引后用正常 query 调用，返回拒答文本，不报错。

    用 try/finally 恢复索引，避免影响后续 9e 等测试。
    """
    clear_index()
    try:
        r = answer(QUESTION)
        assert r.answer == REFUSAL_TEXT
        assert r.citations == []
    finally:
        # 恢复索引：9e 需要索引非空才能走到 _get_client()。
        build_index(indexed)


def test_9e_missing_key(monkeypatch, indexed):
    """契约 9e：未设置 ZHIPU_API_KEY 时调用，抛 ValueError。

    两个关键操作：
      1. monkeypatch.delenv：临时删除环境变量，测试后自动恢复；
      2. _get_client.cache_clear()：清掉可能已缓存的 client，
         否则会复用之前用旧 Key 建好的 client，测不到缺 Key 的分支。
    """
    _get_client.cache_clear()
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    # indexed fixture 保证 search 有结果，answer 才会走到 _get_client()。
    with pytest.raises(ValueError):
        answer(QUESTION)
    # 测试后清理 cache，避免后续测试用到「没有 Key 时建的失败状态」
    # （实际上异常不会被 lru_cache 缓存，但显式清理更清晰）。
    _get_client.cache_clear()


# 契约 9f 为人工检查，不写自动断言：
# 运行 test_9a 后，打印 rag_result.answer 和 citations，
# 人工确认：答案里确实有 [n] 标记；citations 的页码能对应到原文；
# 引用片段和答案内容相关。第 3 块验收时已人工执行并通过。