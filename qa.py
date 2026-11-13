"""PaperLens 第 3/6/7 块：检索 + 重排 + 生成 + 引用 + 拒答。

【模块职责】
  把「用户问题」变成「有出处的答案」。内部流程：
    问题 -> 第 2 块 search() 取 top-20 候选 -> 第 6 块 rerank() 精排取 top-4
         -> 第 7 块 阈值拒答判断 -> 拼 prompt -> 调 GLM-4.7-Flash
         -> 解析 [n] -> 返回 RagAnswer
  阶段 3 新增流式入口 answer_stream()，answer() 改为消费它，保证单一逻辑源。
  阶段 4 新增 degraded 事件：重排失败时降级用粗排 top4 继续，不抛异常。
  阶段 5 接入 loguru 结构化日志：一条/请求，含 token 用量，控制台可读 + 文件 JSON。

【在架构中的位置】
  第 3/6/7 块，是 P0 闭环的核心。
    - 上游依赖：index_store.search()（第 2 块）、rerank.rerank()（第 6 块）。
    - 下游被依赖：第 4 块 Gradio 界面调用 answer_stream() 做流式展示；
      evaluate.py 等脚本调用 answer()（非流式）。
    - 第 7 块评估脚本会调用本模块，衡量 Faithfulness 与拒答准确率。

【遵循的契约】
  第 3 块契约：
    - 4：检索为空时不调 LLM，直接返回 REFUSAL_TEXT。
    - 5：system 用英文写规则，要求同语言作答。
    - 6：引用解析降级策略——解析不到编号时返回全部检索结果。
    - 7：Key 缺失抛 ValueError；API 失败抛异常不吞；本块不打错误堆栈日志。
  第 6 块契约：
    - 5：answer() 内部改为 search(top_k=20) → rerank(final_k=4) → 拼 prompt。
    - 对外签名保持 answer(query, top_k=20, final_k=4)。
  第 7 块契约：
    - 4：重排后若 results[0].score < 阈值，直接返回 REFUSAL_TEXT，不调 LLM。
    - 阈值从 thresholds.json 读取，按语言分别设。缺失抛异常。
    - 语言检测：query 含中文字符视为 zh，否则 en。
    - 阈值优先级：lang 精确匹配 > default。
  阶段 3 契约：
    - 3.2：answer_stream() yield 三种事件（token / citations / refused）。
    - 3.3：answer() 签名与 RagAnswer 结构不变，内部消费 answer_stream()。
    - 3a/3b：空 query / 空索引 → yield refused(reason="empty")。
    - 3c：阈值拒答 → yield refused(reason="below_threshold")。
    - 3d：正常 → 多个 token + 一个 citations。
    - 3e：Key 缺失抛 ValueError（在 _get_client 阶段）。
    - 缺口 B：流式首 token 前可重试，已出 token 后不重试（避免重复输出）。
  阶段 4 契约：
    - 4.2 D1：rerank 抛异常 → yield degraded(rerank_failed) + 用 coarse_results[:4] 继续。
    - 4.2 D5：rerank 返回空 → yield refused(empty)（防御分支）。
    - 4.4：degraded 必为流第一事件、与 refused 互斥、之后仍发 citations。
    - 4.4：fallback_count = min(final_k, len(coarse_results))。
    - 4.3：D1 跳过阈值判断（C3 取舍，量纲不同）。
    - 4.9：不引入新异常类型；真异常仍抛。
  阶段 5 契约：
    - 5.2：loguru，控制台人类可读 + 文件 JSON（logs/app.log，按天轮转留 7 天）。
    - 5.3：INFO 默认；DEBUG 才记 query 原文。
    - 5.4：一条/请求汇总日志，字段含 lang/query_len/retrieval_ms/rerank_ms/llm_ms/
      total_ms/n_candidates/top_score/refused/refuse_reason/degraded/degrade_reason/
      prompt_tokens/completion_tokens/total_tokens。
    - 5.6：_stream_llm yield 元组（"token"/"usage"）；_call() 加 stream_options。
    - 5.7：5g 平台不支持 stream_options 时降级（usage=null 不报错）；
      5h usage 只在流正常结束时出现。
    - 5.9：不把 usage 通过 answer_stream 暴露给 app.py；RagAnswer.usage 保持 None。
  阶段 5 设计决策：
    - D2：logger.bind(**fields).info(msg)，字段进 JSON 的 extra。
    - D3：汇总日志一条/请求，不是每步一条。
    - D5：RagAnswer.usage 保持 None。
    - D6：stream_options 不支持时降级。

【历史修改痕迹】
  - 第 3 块（初始版本）：
    * 创建 Citation、RagAnswer、answer()。
    * 超时 60 秒 → 120 秒（免费档中文长句偶尔超 60 秒）。
  - 第 6 块：
    * 引入 rerank()，answer() 内部改为两阶段检索。
    * top_k 默认值从 8 改为 20，final_k 默认 4。
    * 新增性能日志（检索 / 重排 / 拼 prompt / GLM / 总耗时）。
  - 第 7 块（第一轮）：
    * 新增 REFUSAL_THRESHOLD 常量，从 thresholds.json 读取单值。
    * answer() 在重排后、拼 prompt 前增加阈值判断。
    * 拒答时打印日志「最高分 X < 阈值 Y，拒答」。
  - 第 7 块（第二轮）：
    * thresholds.json 从单值改为 {default, en, zh} 字典，支持按语言设阈值。
    * 新增 _detect_lang()：用中文字符正则判断 query 语言。
    * 新增 _get_threshold()：按 lang 选阈值，缺 lang 时用 default。
    * 拒答日志加上语言标记，便于排查是哪种语言触发的拒答。
    * 兼容旧格式：若 thresholds.json 里 reject 是数字而非字典，自动包成 {"default": 数字}。
  - 阶段 2（已回滚）：
    * qa.py 曾将 search 改为 search_hybrid 别名；实测单文档无增量后回滚为纯向量。
  - 阶段 3（本轮）：
    * 新增 answer_stream()：流式入口，yield token/citations/refused 三种事件。
    * 新增 _stream_llm()：封装 stream=True 调用 + 首 token 前限流重试。
    * answer() 改为消费 answer_stream()，签名与返回结构不变。
    * usage 恒为 None（流式下拿不到完整 usage；阶段 5 起只进日志，仍不改返回结构）。
  - 阶段 3（类型修正）：
    * _stream_llm 的 _call() 标注 -> Any。zhipuai 类型 stub 把 stream=True 的返回推断为
      tuple[str, Any]，与实际 Stream[ChatCompletionChunk] 不符，导致 chunk.choices 报
      「未知属性」。这与 C10/C18 同源（stub 未按 stream 参数重载），运行时行为经测试验证。
  - 阶段 4（本轮）：
    * answer_stream() 的重排段改为 try/except：抛异常 → yield degraded(rerank_failed)。
    * 新增 degraded 事件（第四种）；fallback_count = min(final_k, len(coarse_results))。
    * D1 降级路径跳过阈值判断（C3 取舍：粗排余弦分与重排分不同量纲）。
    * rerank 返回空 → yield refused(empty)（D5 防御分支）。
    * answer() 消费 stream 时遇 degraded 静默忽略（非流式调用方对降级透明）。
  - 阶段 5（本轮）：
    * 接入 loguru：_setup_logging() 幂等配置控制台 + 文件双 sink。
    * _stream_llm 的 yield 从 str 改为 tuple[str, Any]（"token"/"usage"）。
    * _call() 加 stream_options={"include_usage": True}；不支持时降级。
    * answer_stream 消费元组、收集 usage、流结束后打一条汇总日志。
    * 删除 5 处 print("[性能日志]...")，信息并入汇总日志。
    * 拒答/降级路径也打汇总日志（llm_ms/usage 为 null）。
  - 阶段 5（修正）：
    * 新增 _supports_stream_options()：模块加载时探测一次 SDK 是否支持 stream_options。
      zhipuai 2.1.5.20250825 不支持该参数，直接传会抛 TypeError；原先用"试错→降级"
      导致每次请求多打一次被拒请求 + WARNING。改为探测后不支持就直接不带。
    * 运行时降级分支只匹配明确的 stream_options / include_usage 关键词，
      去掉过宽的 "unknown" 条件（避免把无关错误也降级）。

【对外暴露】
  - answer(query, top_k=20, final_k=4) -> RagAnswer ：非流式主入口。
  - answer_stream(query, top_k=20, final_k=4) -> Iterator[dict] ：流式入口。
  - RagAnswer / Citation ：返回结构，供上层序列化或展示。
  - logger ：loguru 实例，app.py 复用以记异常日志。

【依赖】
  - zhipuai ：智谱官方 SDK，调 GLM-4.7-Flash。
  - httpx ：用于设置更精细的超时参数（连接超时 + 读取超时）。
  - loguru ：结构化日志（阶段 5 新增）。
  - index_store ：复用第 2 块的向量检索能力。
  - rerank ：复用第 6 块 Cross-Encoder 重排能力。
"""
import inspect
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import httpx
from loguru import logger
from zhipuai import ZhipuAI

# 【阶段 2 回滚】BM25 混合检索在单文档场景实测无增量（diag_hybrid_report.json），
# 已回滚为纯向量检索。hybrid_search.py 保留作为多文档场景储备。
from index_store import search
from rerank import rerank

# ---------------------------------------------------------------------------
# 阶段 5：日志初始化（幂等）
# ---------------------------------------------------------------------------
_LOG_CONFIGURED = False


def _setup_logging() -> None:
    """幂等配置 loguru：控制台人类可读 + 文件 JSON 双 sink。

    【核心逻辑】
      - 移除 loguru 默认的 stderr sink，换成我们自己配的格式；
      - 控制台 format 只显示时间/级别/message（人类可读）；
      - 文件 sink serialize=True，每条日志是 JSON，bind 的字段在 record.extra 里；
      - rotation="00:00" 按天轮转，retention="7 days" 保留 7 天。

    【边界】
      - logs/ 目录不存在时自动创建（Path.mkdir(exist_ok=True)）；
      - 幂等：_LOG_CONFIGURED 为 True 时直接返回，避免重复加 sink。
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    logger.remove()  # 移除 loguru 默认 sink（默认输出到 stderr，格式不可控）
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/app.log",
        level="INFO",
        serialize=True,       # 文件里每条是 JSON
        rotation="00:00",     # 每天 0 点轮转
        retention="7 days",   # 保留 7 天
        encoding="utf-8",
    )
    _LOG_CONFIGURED = True


_setup_logging()

# ---------------------------------------------------------------------------
# 模块级常量：集中在这里，避免魔法值散落在函数里。
# ---------------------------------------------------------------------------
MODEL_NAME = "glm-4.7-flash"   # 契约：智谱免费模型
TEMPERATURE = 0.1              # 契约：低温度，减少胡编
MAX_TOKENS = 1024              # 契约：答案通常一两百字，1024 足够
# 从 60 秒放宽到 120 秒（第 3 块历史修改）。GLM 免费档在高峰期生成中文长句时，
# 可能会超过 60 秒，导致 httpx.ReadTimeout。
TIMEOUT_SECONDS = 120
RETRY_DELAY_SECONDS = 10       # 契约：遇限流等 10s 重试一次
SNIPPET_LENGTH = 200           # 契约：引用片段截取 200 字符

# 契约 4：检索为空时的固定拒答文本。集中定义，测试和实现共用同一份。
REFUSAL_TEXT = "根据当前论文库，我无法回答这个问题。"

# 第 7 块：从 thresholds.json 读取拒答阈值字典。
# 兼容两种格式：
#   旧格式 {"reject": 0.3} -> 自动包成 {"default": 0.3}
#   新格式 {"reject": {"default": 0.3, "en": 0.3, "zh": 0.18}}
_THRESHOLDS_PATH = "thresholds.json"
if not os.path.exists(_THRESHOLDS_PATH):
    raise FileNotFoundError(f"缺少 {_THRESHOLDS_PATH}，请先创建")
with open(_THRESHOLDS_PATH, "r", encoding="utf-8") as _f:
    _raw_reject = json.load(_f)["reject"]
if isinstance(_raw_reject, dict):
    REFUSAL_THRESHOLDS: dict = dict(_raw_reject)
else:
    REFUSAL_THRESHOLDS = {"default": float(_raw_reject)}
# 兜底：确保 default 存在
if "default" not in REFUSAL_THRESHOLDS:
    REFUSAL_THRESHOLDS["default"] = 0.3

# 第 7 块：中文字符正则，用于判断 query 语言。
# 覆盖 CJK 统一表意文字的基本区（0x4E00-0x9FFF），足够应对中英区分。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _detect_lang(text: str) -> str:
    """判断 query 语言：含中文字符返回 "zh"，否则返回 "en"。

    为什么用正则而不是 langdetect 库：
      契约要求最小实现；本项目只区分中英，一个正则足够，
      避免引入额外依赖。混合语言（中英混杂）时只要含中文就按 zh 处理。
    """
    return "zh" if _CJK_RE.search(text) else "en"


def _get_threshold(lang: str) -> float:
    """按语言取阈值；lang 不在字典里时回退到 default。"""
    return REFUSAL_THRESHOLDS.get(lang, REFUSAL_THRESHOLDS["default"])


def _log_query(
    *,
    query: str,
    lang: str,
    retrieval_ms: float | None,
    rerank_ms: float | None,
    llm_ms: float | None,
    total_ms: float,
    n_candidates: int | None,
    top_score: float | None,
    refused: bool,
    refuse_reason: str | None,
    degraded: bool,
    degrade_reason: str | None,
    usage: Any,
) -> None:
    """打一条汇总日志（阶段 5 / D3：一条/请求，不是每步一条）。

    【参数与字段】
      所有参数对应契约 5.4 的字段。usage 是 zhipuai 的 usage 对象或 None；
      从中取 prompt_tokens / completion_tokens / total_tokens，取不到为 None。

    【隐私（L3）】
      默认 INFO 级只记 query_len，不记 query 原文。
      DEBUG 级才记 query 原文——通过 logger.bind(query=query).debug(...) 实现。

    【核心逻辑】
      用 logger.bind(**fields).info("query done")：bind 的字段进 JSON 的 record.extra，
      可按字段过滤；控制台只显示 message（"query done"）。
    """
    pt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    ct = getattr(usage, "completion_tokens", None) if usage is not None else None
    tt = getattr(usage, "total_tokens", None) if usage is not None else None

    fields = dict(
        event="query",
        lang=lang,
        query_len=len(query),
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        n_candidates=n_candidates,
        top_score=top_score,
        refused=refused,
        refuse_reason=refuse_reason,
        degraded=degraded,
        degrade_reason=degrade_reason,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
    )
    logger.bind(**fields).info("query done")

    # L3：query 原文只在 DEBUG 级记。
    if logger.level("DEBUG").no >= logger.level("INFO").no:
        # 仅当 DEBUG 启用时才 bind 原文（避免无谓开销）。
        logger.bind(event="query_text", query=query).debug("query text")


@dataclass
class Citation:
    """一条引用：指向某个 chunk 的出处。

    为什么需要它（而不是直接返回 Chunk）：
      上层（前端、评估）只需要「页码 + 一小段原文」来展示，
      不需要完整 chunk（可能上千字符）。用 Citation 做投影，
      既减少传输量，也把「引用」和「检索结果」两种概念解耦。

    字段：
      chunk_id     : 对应 chunk 的唯一 ID，便于日志追溯和评估标注。
      doc_name     : 文档文件名，引用面板显示「出自哪篇论文」。
      page_start   : 起始页码（从 1 开始）。
      page_end     : 结束页码；跨页时 > page_start。
      text_snippet : 原文片段，截取前 SNIPPET_LENGTH 个字符。
    """
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    text_snippet: str


@dataclass
class RagAnswer:
    """一次 RAG 问答的完整返回。

    字段：
      answer    : LLM 生成的答案文本；拒答时是 REFUSAL_TEXT。
      citations : 引用列表；拒答时为空列表。
      usage     : token 用量 dict（prompt_tokens / completion_tokens / total_tokens）。
                  P0 若 API 未返回 usage，则为 None；P2 做日志时直接可用。
                  【阶段 3 变更】流式下 answer() 恒为 None（见文件头历史修改痕迹）。
                  【阶段 5 变更】token 统计只进日志，RagAnswer.usage 保持 None（D5）。
    """
    answer: str
    citations: list[Citation]
    usage: dict | None


@lru_cache(maxsize=1)
def _get_client() -> ZhipuAI:
    """创建并缓存智谱客户端，进程内只建一次。

    为什么缓存：
      Client 内部持有 HTTP 连接池，复用比每次新建高效；
      评估脚本会连续调用几十次，复用能省下反复建连的开销。

    错误语义（契约 7）：
      ZHIPU_API_KEY 未设置时，不抛 KeyError，而是抛带明确提示的 ValueError，
      方便上层直接展示给用户「请先配置 API Key」。

    超时策略：
      使用 httpx.Timeout 分别控制连接和读取。
      - connect=10.0：建立 TCP 连接最多等 10 秒，网络断了快速报错。
      - timeout=120.0：等待服务器返回内容的读取超时设为 120 秒。
      这样既不会被瞬时断网卡死，又能给生成较慢的中文答案留足时间。
    """
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        # 契约 7：Key 缺失抛 ValueError，不吞。
        raise ValueError(
            "ZHIPU_API_KEY 未设置。请在终端执行 "
            '$env:ZHIPU_API_KEY = "你的key" 后重试。'
        )
    return ZhipuAI(
        api_key=api_key,
        timeout=httpx.Timeout(timeout=TIMEOUT_SECONDS, connect=10.0),
    )


def _to_citation(retrieved) -> Citation:
    """把第 2 块的 RetrievedChunk 投影成 Citation。

    参数：
      retrieved : index_store.RetrievedChunk，含 chunk 和 score。
                  本函数只用 chunk，score 在引用展示里不需要。

    返回：
      Citation，text_snippet 截取前 SNIPPET_LENGTH 字符。
    """
    c = retrieved.chunk
    # 【边界】用切片而不是 [:SNIPPET_LENGTH] 的边界判断：
    #   Python 切片越界不报错，原文不足 200 字符时自然返回全文，无需 if。
    return Citation(
        chunk_id=c.chunk_id,
        doc_name=c.doc_name,
        page_start=c.page_start,
        page_end=c.page_end,
        text_snippet=c.text[:SNIPPET_LENGTH],
    )


def _build_prompt(query: str, results: list) -> tuple[str, str]:
    """拼出 system 和 user 两条消息。

    参数：
      query   : 用户原问题（可能是中文或英文，不翻译）。
      results : 经过重排后精选出的 RetrievedChunk 列表（通常是 top-4）。
                阶段 4 起，降级路径传入的是粗排 top4，同样兼容。

    返回：
      (system_text, user_text)。

    【核心逻辑】资料格式：
      [1] (page 5)
      <chunk 原文>

      [2] (page 2)
      <chunk 原文>
      ...
      Question: <用户原问题>

    为什么这么拼（设计决策 4）：
      编号是 LLM 标注 [n] 的依据，页码让 LLM 能在答案里体现出处，
      纯文本格式对 LLM 的引用标注比 JSON 更稳、token 更少。
    """
    # system 用英文写规则：LLM 对英文指令的遵循度通常更稳定。
    # 第 4 条是关键：要求 LLM 用与问题相同的语言作答（契约 5 修订版）。
    system_text = (
        "You are a helpful academic paper assistant.\n"
        "Rules:\n"
        "1. Answer the question ONLY based on the provided context. "
        "Do not use outside knowledge.\n"
        "2. If the context does not contain the answer, say clearly that "
        "the provided materials do not cover it.\n"
        "3. Cite the context segments you use with bracket numbers "
        "like [1], [2].\n"
        "4. Answer in the same language as the user's question."
    )

    # 逐条拼资料：编号从 1 开始，与 LLM 输出里的 [n] 对应。
    parts = []
    for i, r in enumerate(results, start=1):
        # 用 page_start 标注页码；跨页 chunk 的 page_end 在引用面板里展示，
        # prompt 里只给起始页，避免信息过载。
        parts.append(f"[{i}] (page {r.chunk.page_start})\n{r.chunk.text}")
    context = "\n\n".join(parts)

    # 问题原样附在末尾，不翻译：LLM 靠「原问题的语言」决定用什么语言回答。
    user_text = f"Context:\n{context}\n\nQuestion: {query}"
    return system_text, user_text


def _parse_citations(answer_text: str, results: list) -> list[Citation]:
    """解析答案里的 [n] 标记，映射回重排后的检索结果。

    参数：
      answer_text : LLM 返回的答案文本。
      results     : 本次重排后输入的 RetrievedChunk 列表（降级时是粗排 top4）。

    返回：
      Citation 列表，按答案中编号首次出现的顺序；无重复。

    【核心逻辑 / 降级策略（契约 6）】
      - 用正则找所有 [数字]；
      - 编号必须落在 1..len(results)，越界忽略；
      - 去重，保持首次出现顺序；
      - 如果一个都没解析到，降级返回全部检索结果。
    """
    seen: set[int] = set()
    ordered_indices: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", answer_text):
        idx = int(m.group(1))
        if 1 <= idx <= len(results) and idx not in seen:
            seen.add(idx)
            ordered_indices.append(idx)

    if not ordered_indices:
        # 契约 6 降级：解析不到任何编号，返回全部重排结果。
        return [_to_citation(r) for r in results]

    return [_to_citation(results[i - 1]) for i in ordered_indices]


def _is_rate_limit_error(err: Exception) -> bool:
    """判断异常是否属于「限流 / 过载」类，用于决定是否重试。

    为什么用字符串匹配而不是 SDK 具体异常类：
      契约要求「最小实现」；zhipuai 的错误类名可能随版本变化，
      匹配错误码（1302 用户限流、1305 平台过载）和常见关键词更稳。
    代价：理论上可能误判，但只在异常消息里命中这些词时才重试，风险可控。
    """
    msg = str(err).lower()
    keywords = ["1302", "1305", "429", "rate limit", "too many requests"]
    return any(k in msg for k in keywords)


def _supports_stream_options() -> bool:
    """探测当前 zhipuai SDK 的 create() 是否接受 stream_options 参数（阶段 5）。

    【为什么需要】
      zhipuai 2.1.5.20250825 的 create() 不接受 stream_options，直接传会抛
      TypeError。若用"试错 → 捕获 → 降级"的写法，每次请求都会多打一次被拒的
      请求 + 打一条 WARNING，既浪费又吵。改为模块加载时探测一次，不支持就直接
      不带该参数。

    【探测方式】
      用 inspect.signature 看 Completions.create 的参数列表里有没有
      stream_options。失败（SDK 结构变化）时保守返回 False（不带 usage）。

    【边界】
      这是一次性探测，结果不缓存到模块级变量也行——inspect 本身很便宜。
      但如果将来想缓存，加 @lru_cache(maxsize=1) 即可。
    """
    try:
        from zhipuai.api_resource.chat.completions import Completions
        sig = inspect.signature(Completions.create)
        return "stream_options" in sig.parameters
    except Exception:
        # SDK 结构不可预期时保守返回 False：不带 usage，不报错。
        return False


def _stream_llm(client: ZhipuAI, messages: list) -> Iterator[tuple[str, Any]]:
    """流式调用 GLM，逐段 yield 元组 ("token", text) 或 ("usage", usage_obj)。

    【方法契约（阶段 5 变更）】
      返回：Iterator[tuple[str, Any]]。
        - ("token", str)：本次增量文本（非空）。
        - ("usage", usage_obj)：流结束时最后一个 chunk 带的 usage 对象（可能为 None）。
      为什么改成元组：usage 在最后一个 chunk 里，需要传出去给 answer_stream 打日志；
      它属于内部协议，answer_stream 的对外 yield 协议不变（仍只 yield token/citations/
      refused/degraded 四种事件）。

    【错误语义】
      Key 缺失在 _get_client() 阶段已抛；API 失败抛异常，不吞。

    【缺口 B：首 token 前可重试，已出 token 后不重试】
      原因：流式的输出不可撤回——已 yield 出去的 token 收不回。若中途限流后重试，
      会把答案前半段重复输出。所以：
        - 首 token 之前遇限流 → 安全重试一次（此时什么都没输出）。
        - 已出首 token 之后遇限流 → 直接抛，由上层处理（不重复）。
      限流通常发生在「请求发起时」而非「生成中途」，所以首 token 前重试覆盖了大部分场景。

    【阶段 4 / D2】首 token 前重试仍失败 → 抛异常，由 app.py 转成「生成服务繁忙」提示。

    【阶段 5 / 5g】stream_options={"include_usage": True}：请求最后一个 chunk 带 usage。
      探测（_supports_stream_options）在模块加载时做一次：不支持就直接不带该参数，
      避免每次请求多打一次被拒的请求。运行时若仍被拒（探测误判），降级一次。

    【性能隐患】
      流式响应本身不占额外内存；但 chunk 迭代是网络 IO，单次生成 6-10 秒。
    """
    def _call(include_usage: bool = True) -> Any:
        # 契约 3.3：与 answer() 一致的参数。stream=True 是流式开关。
        # 【假设 / 已知缺陷】zhipuai 的类型 stub 没有按 stream 参数做重载，
        # 把 stream=True 的返回错误推断为 tuple[str, Any]，运行时实际是
        # Stream[ChatCompletionChunk]（有 .choices）。这里显式标注 -> Any，
        # 表示「此返回值的类型检查不可靠，运行时行为已用测试验证」。
        # 与 C10 / C18 同源（stub 未重载）。
        kwargs: dict[str, Any] = dict(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        # 阶段 5：只有 SDK 支持且调用方要 usage 时才传 stream_options。
        if include_usage and _supports_stream_options():
            kwargs["stream_options"] = {"include_usage": True}
        return client.chat.completions.create(**kwargs)

    first_token_seen = False

    def _iter_chunks(resp: Any) -> Iterator[tuple[str, Any]]:
        """迭代流式 chunk，yield ("token"/"usage", value)。"""
        for chunk in resp:
            # 【假设】流式 chunk 的 choices[0].delta.content 可能为 None（首包/尾包），
            # 用 `or ""` 兼容，空串不 yield。
            choices = getattr(chunk, "choices", None)
            if choices:
                delta = choices[0].delta.content or ""
                if delta:
                    yield ("token", delta)
            # 阶段 5：最后一个 chunk 带 usage（可能为 None，取决于平台）。
            u = getattr(chunk, "usage", None)
            if u is not None:
                yield ("usage", u)

    try:
        try:
            resp: Any = _call(include_usage=True)
        except Exception as e:
            # 5g 兜底：探测误判时（比如 SDK 支持但平台拒），运行时降级一次。
            # 只匹配明确的 stream_options 关键词，避免把无关错误也降级。
            msg = str(e).lower()
            if "stream_options" in msg or "include_usage" in msg:
                logger.warning("stream_options 被拒，降级为不带 usage 重试：{}", e)
                resp = _call(include_usage=False)
            else:
                raise
        for item in _iter_chunks(resp):
            if item[0] == "token":
                first_token_seen = True
            yield item
    except Exception as e:
        # 【缺口 B】只在首 token 前重试：此时还没 yield 任何东西，重试安全。
        if not first_token_seen and _is_rate_limit_error(e):
            logger.info("流式首 token 前触发限流，等待 {} 秒后重试", RETRY_DELAY_SECONDS)
            time.sleep(RETRY_DELAY_SECONDS)
            resp = _call(include_usage=True)
            for item in _iter_chunks(resp):
                if item[0] == "token":
                    first_token_seen = True
                yield item
        else:
            # 首 token 后限流 / 非限流错误：直接抛，不重试（避免重复输出）。
            raise


def answer_stream(
    query: str, top_k: int = 20, final_k: int = 4
) -> Iterator[dict]:
    """RAG 流式主入口：检索 + 重排 + 拒答判断 + 流式生成 + 最后一次性给引用。

    【方法契约】
      参数：与 answer() 一致。
      返回：Iterator[dict]，四种事件：
        {"type": "token", "content": "..."}       LLM 每段增量
        {"type": "citations", "citations": [...]} 流结束后一次性给引用
        {"type": "refused", "reason": "empty"|"below_threshold"}  拒答，无 token
        {"type": "degraded", "reason": "rerank_failed", "fallback_count": <int>}
                                                  重排失败，已降级为粗排
      错误：Key 缺失抛 ValueError；API 失败抛异常，不吞（与 answer() 一致）。

    【阶段 4 / D1】rerank 抛异常 → yield degraded，用 coarse_results[:final_k] 继续。
      - degraded 是流的第一事件（在第一个 token 之前）。
      - fallback_count = min(final_k, len(coarse_results))。
      - 降级路径跳过阈值判断（C3 取舍：粗排余弦分与重排分不同量纲）。

    【阶段 4 / D5】rerank 正常返回但列表空 → yield refused(empty)（防御分支）。

    【阶段 5】流结束后打一条汇总日志（含 token 用量）。usage 不从本函数 yield 出去
      （不暴露给 app.py），只进日志（D5）。
    """
    t_start = time.time()

    # 步骤 1-2：语言检测 + 阈值选择。
    lang = _detect_lang(query)
    threshold = _get_threshold(lang)

    # 步骤 3：向量粗排。
    t0 = time.time()
    coarse_results = search(query, top_k=top_k)
    retrieval_ms = (time.time() - t0) * 1000
    n_candidates = len(coarse_results)

    # 步骤 4（契约 3a / D4）：检索为空 -> 不调 LLM，直接拒答。
    if not coarse_results:
        _log_query(
            query=query, lang=lang,
            retrieval_ms=retrieval_ms, rerank_ms=None, llm_ms=None,
            total_ms=(time.time() - t_start) * 1000,
            n_candidates=n_candidates, top_score=None,
            refused=True, refuse_reason="empty",
            degraded=False, degrade_reason=None, usage=None,
        )
        yield {"type": "refused", "reason": "empty"}
        return

    # 步骤 5（阶段 4 / D1 + D5）：Cross-Encoder 精排，失败则降级。
    t1 = time.time()
    skip_threshold = False
    degraded = False
    degrade_reason: str | None = None
    try:
        fine_results = rerank(query, coarse_results, final_k=final_k)
    except Exception as e:
        # D1：重排失败 → 降级为粗排 top final_k，用 degraded 事件告知上层。
        # 这是"设计内可预期失败"，不是异常；真异常仍会走到下面的 raise。
        fallback = coarse_results[:final_k]
        yield {
            "type": "degraded",
            "reason": "rerank_failed",
            "fallback_count": min(final_k, len(fallback)),
        }
        results = fallback
        # 已知取舍 C3：降级路径不判阈值——粗排余弦分与重排 Cross-Encoder 分量纲不同。
        skip_threshold = True
        degraded = True
        degrade_reason = "rerank_failed"
        rerank_ms = None
        logger.warning("重排失败，降级为粗排 top{}: {!r}", len(fallback), e)
    else:
        # D5（防御分支）：rerank 正常返回但列表空，等同检索为空。
        if not fine_results:
            _log_query(
                query=query, lang=lang,
                retrieval_ms=retrieval_ms, rerank_ms=(time.time() - t1) * 1000,
                llm_ms=None, total_ms=(time.time() - t_start) * 1000,
                n_candidates=n_candidates, top_score=None,
                refused=True, refuse_reason="empty",
                degraded=False, degrade_reason=None, usage=None,
            )
            yield {"type": "refused", "reason": "empty"}
            return
        results = fine_results
        rerank_ms = (time.time() - t1) * 1000

    # 步骤 7（契约 3c / D4）：阈值拒答判断。仅在重排成功时执行。
    top_score: float | None = None
    if not skip_threshold:
        top_score = results[0].score
        if top_score < threshold:
            _log_query(
                query=query, lang=lang,
                retrieval_ms=retrieval_ms, rerank_ms=rerank_ms, llm_ms=None,
                total_ms=(time.time() - t_start) * 1000,
                n_candidates=n_candidates, top_score=top_score,
                refused=True, refuse_reason="below_threshold",
                degraded=False, degrade_reason=None, usage=None,
            )
            yield {"type": "refused", "reason": "below_threshold"}
            return

    # 步骤 8：拼 prompt（函数化，降级路径传入粗排 results 同样兼容 —— T2 单一源）。
    system_text, user_text = _build_prompt(query, results)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    # 步骤 9（契约 3e）：流式调 LLM。Key 缺失时 _get_client() 抛 ValueError。
    t3 = time.time()
    client = _get_client()
    full_text_parts: list[str] = []
    usage: Any = None
    for kind, value in _stream_llm(client, messages):
        if kind == "token":
            full_text_parts.append(value)
            yield {"type": "token", "content": value}
        elif kind == "usage":
            # 阶段 5：usage 只进日志，不 yield 给 app.py（D5）。
            usage = value
    llm_ms = (time.time() - t3) * 1000

    # 步骤 10（契约 3d / T4）：流结束后一次性解析引用。
    full_text = "".join(full_text_parts)
    citations = _parse_citations(full_text, results)
    yield {"type": "citations", "citations": citations}

    # 阶段 5：汇总日志（含 token 用量）。
    _log_query(
        query=query, lang=lang,
        retrieval_ms=retrieval_ms, rerank_ms=rerank_ms, llm_ms=llm_ms,
        total_ms=(time.time() - t_start) * 1000,
        n_candidates=n_candidates, top_score=top_score,
        refused=False, refuse_reason=None,
        degraded=degraded, degrade_reason=degrade_reason,
        usage=usage,
    )


def answer(query: str, top_k: int = 20, final_k: int = 4) -> RagAnswer:
    """RAG 主入口（非流式）：检索 + 重排 + 拒答判断 + 生成 + 引用。

    参数：
      query   : 用户问题。
      top_k   : 向量检索的粗排候选数，默认 20（第 6 块修改）。
      final_k : 重排后送入 LLM 的段落数，默认 4（第 6 块修改）。

    返回：
      RagAnswer。检索为空或最高分低于阈值时返回拒答文本 + 空引用。

    错误语义（契约 7）：
      - Key 缺失：_get_client() 抛 ValueError，本函数不捕获。
      - API 调用失败：由 _stream_llm 处理（首 token 前限流重试一次；否则抛）。
      本函数不吞异常、不包装、不打错误堆栈日志（业务指标进汇总日志）。

    【算法步骤（阶段 3/4/5 版）】
      内部消费 answer_stream()：
        收到 token    → 追加到 tokens 缓冲
        收到 citations→ 记录引用
        收到 refused  → 直接返回 REFUSAL_TEXT + 空引用
        收到 degraded → 静默忽略（非流式调用方对降级透明，阶段 6 日志补可观测性）
      结束后返回 RagAnswer(answer="".join(tokens), citations=..., usage=None)。

    【为什么 degraded 被忽略】
      阶段 4 契约 20f：answer() 消费 stream 时遇 degraded 必须忽略该事件、正常聚合
      token、不崩。降级信息对非流式调用方不可见，是"不改 RagAnswer 结构"的必然代价。

    【阶段 5 / D5】usage 恒为 None：token 统计只进日志，不改返回结构。
    """
    tokens: list[str] = []
    citations: list[Citation] = []
    for event in answer_stream(query, top_k=top_k, final_k=final_k):
        etype = event["type"]
        if etype == "token":
            tokens.append(event["content"])
        elif etype == "citations":
            citations = event["citations"]
        elif etype == "refused":
            # 契约 4：拒答时返回固定文本 + 空引用。
            return RagAnswer(answer=REFUSAL_TEXT, citations=[], usage=None)
        # degraded：静默忽略（见 docstring）。
    return RagAnswer(answer="".join(tokens), citations=citations, usage=None)