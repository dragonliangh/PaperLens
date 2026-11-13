"""第 3 块验收测试：对应契约 9a-9f + 阶段 3 流式契约 19a-19c + 阶段 4 容错契约 20a-20g。

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
  - 阶段 3 新增 19a（需 Key）、19b/19c（不需 Key）。
  - 阶段 4 新增 20a-20g：全部 mock rerank/_stream_llm，不真实调 LLM，不需要 Key。

【历史修改痕迹】
  - 阶段 3：新增 answer_stream 测试 3 个（19a/19b/19c）。
  - 阶段 4：新增容错测试 7 个（20a-20g）；20a/20b/20d/20f/20g 用 monkeypatch
    注入，不依赖 API Key，可进 CI（但依赖 Chroma indexed，本地跑）。
  - 阶段 5：_stream_llm 的 yield 从 str 改为 tuple[str, Any]；
    _mock_stream_tokens 和 test_20d 的 _fake_stream 同步改为 yield 元组，
    否则 answer_stream 解包会报 ValueError。
"""
import os
import time

import pytest

from ingest import parse_and_chunk
from index_store import build_index, clear_index, Chunk, RetrievedChunk
from qa import REFUSAL_TEXT, _get_client, answer, answer_stream

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


# ===========================================================================
# 阶段 3 流式测试（契约 19a-19c）
# ===========================================================================
@needs_sample
@needs_key
def test_19a_stream_normal(indexed):
    """契约 19a：正常问题 yield 多个 token + 一个 citations，拼接等于完整答案。

    真实调用 LLM，需要 ZHIPU_API_KEY，耗时几秒。
    调用前 sleep 2s，避免与其他 API 调用太密触发限流。
    """
    time.sleep(2)
    events = list(answer_stream(QUESTION))
    types = [e["type"] for e in events]

    # 至少一个 token，最后一个事件必须是 citations，且不出现 refused / degraded。
    assert "token" in types
    assert types[-1] == "citations"
    assert "refused" not in types
    assert "degraded" not in types

    text = "".join(e["content"] for e in events if e["type"] == "token")
    citations = events[-1]["citations"]
    assert text.strip()
    assert len(citations) > 0


@needs_sample
def test_19b_stream_empty(indexed):
    """契约 19b：空 query yield 一条 refused（reason=empty），无 token。

    不调 LLM，不需要 Key。
    """
    events = list(answer_stream(""))
    assert len(events) == 1
    assert events[0]["type"] == "refused"
    assert events[0]["reason"] == "empty"


@needs_sample
def test_19c_stream_refused(indexed):
    """契约 19c：不可答题触发阈值拒答，yield refused（reason=below_threshold），无 token。

    不调 LLM（拒答发生在流开始前），不需要 Key。
    用与 test_eval 相同的不可答题，确保阈值拒答稳定触发。
    """
    events = list(
        answer_stream("What did the authors eat for lunch during the experiments?")
    )
    assert len(events) == 1
    assert events[0]["type"] == "refused"
    assert events[0]["reason"] == "below_threshold"


# ===========================================================================
# 阶段 4 容错测试（契约 20a-20g）
# 全部用 monkeypatch 注入失败，不真实调 LLM，不需要 API Key。
# ===========================================================================
def _mock_stream_tokens(tokens: list[str]):
    """构造一个假的 _stream_llm，按顺序 yield ("token", text) 后正常结束。

    【阶段 5 变更】_stream_llm 的 yield 从 str 改为 tuple[str, Any]，
    mock 必须同步，否则 answer_stream 解包会报 ValueError。
    不 yield ("usage", ...)，模拟"平台没返回 usage"的降级路径。
    """
    def _fake(client, messages):
        for t in tokens:
            yield ("token", t)
    return _fake


def _mock_rerank_raises(exc: Exception):
    """构造一个假的 rerank，调用即抛异常。"""
    def _fake(query, candidates, final_k=4):
        raise exc
    return _fake


def _mock_rerank_ok():
    """构造一个假的 rerank，原样返回前 final_k 个（带原 score）。"""
    def _fake(query, candidates, final_k=4):
        return candidates[:final_k]
    return _fake


@needs_sample
def test_20a_degraded_first_event(indexed, monkeypatch):
    """契约 20a：monkeypatch qa.rerank 抛异常 → degraded 是首事件 + 后续 token/citations。

    20a 的三条断言：
      ① degraded 是事件序列首项（4.4 写死"必在第一个 token 之前"）；
      ② fallback_count = min(4, len(coarse_results))；
      ③ 后续有 token 和 citations。
    """
    import qa
    monkeypatch.setattr(qa, "rerank", _mock_rerank_raises(RuntimeError("mock rerank failure")))
    monkeypatch.setattr(qa, "_stream_llm", _mock_stream_tokens(["Mocked ", "answer ", "[1]"]))

    events = list(qa.answer_stream(QUESTION))
    types = [e["type"] for e in events]

    # ① degraded 是首事件
    assert types[0] == "degraded"
    assert events[0]["reason"] == "rerank_failed"

    # ② fallback_count = min(4, 粗排结果数)；粗排 top20 非空，所以是 1..4
    assert 1 <= events[0]["fallback_count"] <= 4

    # ③ 后续有 token 和 citations
    assert "token" in types
    assert types[-1] == "citations"


@needs_sample
def test_20b_degraded_no_threshold(indexed, monkeypatch):
    """契约 20b：重排失败时不判阈值，不会出现 refused 事件。

    若降级路径误判阈值（拿粗排余弦分和 Cross-Encoder 阈值比），
    可能把可答题判成 below_threshold。本测试确保 D1 路径跳过阈值判断。
    """
    import qa
    monkeypatch.setattr(qa, "rerank", _mock_rerank_raises(RuntimeError("mock")))
    monkeypatch.setattr(qa, "_stream_llm", _mock_stream_tokens(["ok"]))

    events = list(qa.answer_stream(QUESTION))
    types = [e["type"] for e in events]
    assert "refused" not in types
    assert "degraded" in types


@needs_sample
def test_20c_normal_no_degraded(indexed, monkeypatch):
    """契约 20c：正常路径（rerank 成功）无 degraded 事件，行为与阶段 3 一致。"""
    import qa
    monkeypatch.setattr(qa, "rerank", _mock_rerank_ok())
    monkeypatch.setattr(qa, "_stream_llm", _mock_stream_tokens(["Normal ", "[1]"]))

    events = list(qa.answer_stream(QUESTION))
    types = [e["type"] for e in events]
    assert "degraded" not in types
    assert "token" in types
    assert types[-1] == "citations"


@needs_sample
def test_20d_stream_interrupted(indexed, monkeypatch):
    """契约 20d：流中途中断 → app._on_send 保留半截答案 + 中断提示，页面不崩。

    用 monkeypatch 让 _stream_llm 先 yield 一个 token 再 raise，
    确保中断发生在"已出 token 之后"（D3 场景）。
    """
    import qa
    import app

    def _fake_stream(client, messages):
        # 阶段 5：_stream_llm 现在 yield 元组，mock 要同步。
        yield ("token", "Partial ")
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(qa, "rerank", _mock_rerank_ok())
    monkeypatch.setattr(qa, "_stream_llm", _fake_stream)

    outputs = list(app._on_send(QUESTION, []))
    # _on_send 每次 yield (chatbot, history_state, query_box, citations_md)
    last_chatbot, _, _, last_citations = outputs[-1]
    last_msg = last_chatbot[-1]["content"]

    # 半截答案保留
    assert "Partial" in last_msg
    # 中断提示存在（429 → "生成服务繁忙"）
    assert "繁忙" in last_msg or "中断" in last_msg
    # G5：D3 时引用区标注中断
    assert "中断" in last_citations


def test_20e_error_to_message():
    """契约 20e：_error_to_message 把各类异常转成友好文案（纯函数，不需索引）。

    覆盖四种映射 + 兜底。
    """
    import app
    assert "ZHIPU_API_KEY" in app._error_to_message(ValueError("ZHIPU_API_KEY 未设置"))
    assert "繁忙" in app._error_to_message(RuntimeError("429 Too Many Requests"))
    assert "繁忙" in app._error_to_message(RuntimeError("rate limit exceeded"))
    assert "检索" in app._error_to_message(RuntimeError("chroma collection error"))
    assert "暂时不可用" in app._error_to_message(RuntimeError("something unknown"))


@needs_sample
def test_20f_answer_ignores_degraded(indexed, monkeypatch):
    """契约 20f（硬验收）：answer() 消费 stream 遇 degraded 必须忽略、正常聚合 token、不崩。

    模拟 D1：rerank 抛异常 → degraded 事件 → 后续 token + citations。
    answer() 应返回正常 RagAnswer，内容来自 token，不因 degraded 崩溃。
    """
    import qa
    monkeypatch.setattr(qa, "rerank", _mock_rerank_raises(RuntimeError("mock")))
    monkeypatch.setattr(qa, "_stream_llm", _mock_stream_tokens(["Answer ", "text ", "[1]"]))

    r = qa.answer(QUESTION)
    # 不崩、有答案、有引用
    assert r.answer == "Answer text [1]"
    assert len(r.citations) == 1


@needs_sample
def test_20g_fallback_count(indexed, monkeypatch):
    """契约 20g：fallback_count = min(4, 粗排结果数)。

    用 monkeypatch 让 qa.search 返回固定 N 条（N=2 和 N=5 各跑一次），
    断言 degraded.fallback_count == min(4, N)。
    """
    import qa

    def _make_results(n: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=Chunk(
                    text=f"chunk {i}",
                    doc_name="mock.pdf",
                    page_start=1,
                    page_end=1,
                    section="",
                    chunk_id=f"mock#c{i}",
                ),
                score=0.9,
            )
            for i in range(n)
        ]

    for n in (2, 5):
        monkeypatch.setattr(qa, "search", lambda q, top_k=20, _n=n: _make_results(_n))
        monkeypatch.setattr(qa, "rerank", _mock_rerank_raises(RuntimeError("mock")))
        monkeypatch.setattr(qa, "_stream_llm", _mock_stream_tokens(["ok"]))

        events = list(qa.answer_stream(QUESTION))
        assert events[0]["type"] == "degraded"
        assert events[0]["fallback_count"] == min(4, n), (
            f"N={n} 时 fallback_count 应为 {min(4, n)}，实际 {events[0]['fallback_count']}"
        )