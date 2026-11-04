"""PaperLens 第 3/6 块：检索 + 重排 + 生成 + 引用。

【模块职责】
  把「用户问题」变成「有出处的答案」。内部流程：
    问题 -> 第 2 块 search() 取 top-20 候选 -> 第 6 块 rerank() 精排取 top-4
         -> 拼 prompt -> 调 GLM-4.7-Flash -> 解析 [n] -> 返回 RagAnswer

【在架构中的位置】
  第 3/6 块，是 P0 闭环的核心。
    - 上游依赖：index_store.search()（第 2 块）提供粗排结果，
                rerank.rerank()（第 6 块）提供 Cross-Encoder 精排结果。
    - 下游被依赖：第 4 块 Gradio 界面调用本模块的 answer() 展示答案与引用。
    - 第 7 块评估脚本会调用本模块，衡量 Faithfulness 与拒答准确率。

【对外暴露】
  - answer(query, top_k=20, final_k=4) -> RagAnswer ：唯一主入口。
  - RagAnswer / Citation ：返回结构，供上层序列化或展示。

【依赖】
  - zhipuai ：智谱官方 SDK，调 GLM-4.7-Flash。
  - httpx ：用于设置更精细的超时参数（连接超时 + 读取超时）。
  - index_store ：复用第 2 块的向量检索能力。
  - rerank ：复用第 6 块的 Cross-Encoder 重排能力。
"""
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from zhipuai import ZhipuAI

from index_store import search
from rerank import rerank  # 第 6 块新增：引入 Cross-Encoder 精排

# ---------------------------------------------------------------------------
# 模块级常量：集中在这里，避免魔法值散落在函数里。
# ---------------------------------------------------------------------------
MODEL_NAME = "glm-4.7-flash"   # 契约：智谱免费模型
TEMPERATURE = 0.1              # 契约：低温度，减少胡编
MAX_TOKENS = 1024              # 契约：答案通常一两百字，1024 足够
# 从 60 秒放宽到 120 秒。GLM 免费档在高峰期生成中文长句时，
# 可能会超过 60 秒，导致 httpx.ReadTimeout。
TIMEOUT_SECONDS = 120          # 契约：单次调用超时 120s
RETRY_DELAY_SECONDS = 10       # 契约：遇限流等 10s 重试一次
SNIPPET_LENGTH = 200           # 契约：引用片段截取 200 字符

# 契约 4：检索为空时的固定拒答文本。集中定义，测试和实现共用同一份。
REFUSAL_TEXT = "根据当前论文库，我无法回答这个问题。"


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
      results     : 本次重排后输入的 RetrievedChunk 列表（顺序即 [1][2]... 的顺序）。

    返回：
      Citation 列表，按答案中编号首次出现的顺序；无重复。

    【核心逻辑 / 降级策略（契约 6）】
      - 用正则找所有 [数字]；
      - 编号必须落在 1..len(results)，越界忽略（防止 LLM 编出 [9]）；
      - 去重，保持首次出现顺序；
      - 如果一个都没解析到（LLM 没标或格式不对），降级返回「全部检索结果」，
        保证请求不因引用解析失败而失败。
    """
    seen: set[int] = set()
    ordered_indices: list[int] = []
    # finditer 按出现顺序遍历所有匹配。
    for m in re.finditer(r"\[(\d+)\]", answer_text):
        idx = int(m.group(1))
        # 【边界】越界编号忽略：LLM 可能标 [9] 但本次只有 4 条。
        if 1 <= idx <= len(results) and idx not in seen:
            seen.add(idx)
            ordered_indices.append(idx)

    if not ordered_indices:
        # 契约 6 降级：解析不到任何编号，返回全部重排结果。
        return [_to_citation(r) for r in results]

    # 编号是 1-based，results 是 0-based，映射时减 1。
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


def answer(query: str, top_k: int = 20, final_k: int = 4) -> RagAnswer:
    """RAG 主入口：检索 + 重排 + 生成 + 引用。

    参数：
      query   : 用户问题。
      top_k   : 向量检索的粗排候选数，默认 20。传给 index_store.search()。
      final_k : 重排后送入 LLM 的段落数，默认 4。传给 rerank.rerank()。

    返回：
      RagAnswer。检索为空或重排后为空时返回拒答文本 + 空引用。

    错误语义（契约 7）：
      - Key 缺失：_get_client() 抛 ValueError，本函数不捕获。
      - API 调用失败：限流类错误重试一次后仍失败则抛；其他错误直接抛。
      本函数不吞异常、不包装、不打错误堆栈日志（只打性能日志）。

    【算法步骤（第 6 块修改版）】
      1. 调 search() 向量检索 top_k（默认 20）个候选；
      2. 结果为空 -> 直接返回拒答，不调重排和 LLM；
      3. 调 rerank() 对候选进行 Cross-Encoder 精排，取 final_k（默认 4）个；
      4. 拼 system/user 消息；
      5. 调 GLM；限流则等 10s 重试一次；
      6. 解析答案里的 [n] 得到引用；
      7. 读取 usage（若有）；返回 RagAnswer。
    """
    # 步骤 1：向量粗排。注意 search() 内部已处理空 query / 空库，返回 []。
    t0 = time.time()
    results = search(query, top_k=top_k)
    print(f"[性能日志] 1. 向量检索耗时: {time.time() - t0:.2f} 秒")

    # 步骤 2（契约 4）：检索为空 -> 不调 LLM，直接拒答。
    # 【核心逻辑】这是第 7 块阈值拒答的雏形；P0 只处理「结果为空」。
    if not results:
        print("[性能日志] 检索为空，直接拒答，跳过重排与 LLM。")
        return RagAnswer(answer=REFUSAL_TEXT, citations=[], usage=None)

    # 步骤 3（第 6 块核心逻辑）：Cross-Encoder 精排。
    # 将粗排召回的 top_k 个候选放入 Cross-Encoder 中逐对打分，淘汰语义相关
    # 但不直接回答问题的噪声，保留得分最高的 final_k 个。
    t1 = time.time()
    results = rerank(query, results, final_k=final_k)
    print(f"[性能日志] 2. 重排耗时: {time.time() - t1:.2f} 秒")

    # 边界防御：如果重排后意外为空（理论上不会，除非模型异常），直接拒答。
    if not results:
        return RagAnswer(answer=REFUSAL_TEXT, citations=[], usage=None)

    # 步骤 4：拼 prompt。
    t2 = time.time()
    system_text, user_text = _build_prompt(query, results)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    print(f"[性能日志] 3. 构建 Prompt 耗时: {time.time() - t2:.2f} 秒")

    # 步骤 5：调 LLM。把单次调用封成内部函数，便于重试时复用。
    t3 = time.time()
    client = _get_client()

    def _call() -> Any:
        # 【假设 / 已知缺陷】zhipuai 的类型 stub 没有按 stream 参数做重载，
        # 不传流式时运行时返回 ChatCompletion（有 .choices），但 Pylance 仍推断为
        # StreamResponse（无 .choices），导致下游 resp.choices 报「未知属性」。
        # 这里显式标注 -> Any，表示「此返回值的类型检查不可靠，运行时行为已用测试验证」。
        # 不用 cast 到具体 ChatCompletion 的原因：其导入路径依赖 zhipuai 版本，
        # 写死路径反而脆弱；Any 只影响这一个内部函数的静态检查，代价可控。
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            # 显式声明非流式（契约：P0 不用流式）。
            stream=False,
        )

    try:
        resp = _call()
    except Exception as e:
        # 契约 1：限流类错误等 10s 重试一次；其他错误直接抛。
        if _is_rate_limit_error(e):
            print(f"[性能日志] 触发限流，等待 {RETRY_DELAY_SECONDS} 秒后重试...")
            time.sleep(RETRY_DELAY_SECONDS)
            # 【边界】第二次调用不再捕获：仍失败则异常向上抛（契约 7）。
            resp = _call()
        else:
            raise

    print(f"[性能日志] 4. GLM 接口调用耗时: {time.time() - t3:.2f} 秒")

    # 步骤 6：取答案文本并解析引用。
    # 【边界】content 理论上可能为 None（极端情况），用 or "" 兜底，
    # 避免后续正则对 None 报错；空答案时 _parse_citations 会走降级分支。
    answer_text = resp.choices[0].message.content or ""
    citations = _parse_citations(answer_text, results)

    # 步骤 7：读取 usage。P0 能读到就填，读不到填 None。
    usage = None
    resp_usage = getattr(resp, "usage", None)
    if resp_usage is not None:
        usage = {
            "prompt_tokens": getattr(resp_usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp_usage, "completion_tokens", None),
            "total_tokens": getattr(resp_usage, "total_tokens", None),
        }

    print(f"[性能日志] 5. 引用解析与返回总耗时: {time.time() - t0:.2f} 秒")
    return RagAnswer(answer=answer_text, citations=citations, usage=usage)