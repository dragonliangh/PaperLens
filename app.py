"""PaperLens 第 4/5 块：Gradio 界面 + 云端启动适配。

【模块职责】
  把前三块的「命令行能力」包成一个网页：
    上传 PDF -> 建索引 -> 提问 -> 显示答案和引用。
  同时适配本地开发与 ModelScope Docker 云端部署。
  阶段 3 起 _on_send 改为流式（yield），1-2 秒出第一个字。
  阶段 4 起 _on_send 处理 degraded / 中断 / 异常，转成用户友好文案。
  阶段 5 起异常日志走 loguru（复用 qa.logger），业务日志仍在 qa.py。

【在架构中的位置】
  第 4/5 块，是用户接触的最终边界。
    - 上游依赖：ingest、index_store、qa。
    - 下游被依赖：Docker 容器启动时运行本文件。
    - 界面层是「最终用户边界」，有责任把异常转成用户能看懂的提示。

【对外暴露】
  - build_app() -> gr.Blocks ：构建并返回界面对象。

【依赖】
  - gradio ：界面库（已降级至 5.49.1，兼容魔搭前端环境）。
  - ingest / index_store / qa ：复用前三块的业务逻辑。
  - rerank ：启动预热时使用（局部 import，避免模块被 import 时连带加载）。
  - loguru ：复用 qa.logger，记异常日志（阶段 5）。

【历史修改痕迹】
  - 第 4 块（初始版本）：_on_send 一次性返回完整答案。
  - 阶段 2（已回滚）：_on_upload 曾加 rebuild_bm25 配对；实测无增量后移除。
  - 阶段 3（本轮）：_on_send 改为 yield 流式；import answer_stream 替代 answer。
  - 阶段 3（占位）：_on_send 立即 yield "正在检索并重排…"，解决重排 9 秒白屏。
  - 阶段 4（本轮）：
    * 新增 _error_to_message()：异常 → 用户友好文案（F3：文案归界面层）。
    * _on_send 新增 pending_prefix 机制：degraded 不即时渲染，等首 token 到达时拼接。
    * 流中途异常永远 append 而非覆盖现有 content（D1+D2、D1+D3 叠加场景）。
    * 未知事件类型 print 到 stderr（G7），不崩，向前兼容。
    * 引用区：D3（有半截答案）显示"（生成中断，无引用）"（G5）。
  - 阶段 5（本轮）：
    * _error_to_message 调用处加 logger.error 结构化日志（L2：app 只记异常）。
    * 未知事件类型的 print(..., file=sys.stderr) 保留（契约 5e，不属业务日志）。
"""
import os
import sys
import gradio as gr

from ingest import parse_and_chunk
from index_store import build_index, clear_index
from qa import answer_stream, logger, REFUSAL_TEXT

# ---------------------------------------------------------------------------
# 界面文案集中定义，避免散落在组件定义里。
# ---------------------------------------------------------------------------
TITLE = "PaperLens — 学术论文 RAG 助手"
SUBTITLE = "上传论文 PDF，基于原文回答问题，并给出页码级引用。"
EMPTY_CITATIONS = "（无引用）"
INTERRUPTED_CITATIONS = "（生成中断，无引用）"   # G5：D3 时引用区文案
PLACEHOLDER = "正在检索并重排…"                   # 阶段 3：重排 9 秒的占位提示
DEGRADED_PREFIX = "⚠️ 重排暂时不可用，以下基于粗排结果\n\n"   # 阶段 4：D1 降级前缀


def _format_citations(citations) -> str:
    """把 qa.Citation 列表渲染成 Markdown。"""
    if not citations:
        return EMPTY_CITATIONS
    lines = ["**引用：**"]
    for i, c in enumerate(citations, start=1):
        lines.append(
            f"{i}. `{c.doc_name}` p{c.page_start}-{c.page_end} — `{c.text_snippet}`"
        )
    return "\n".join(lines)


def _error_to_message(exc: Exception) -> str:
    """把异常转成用户能看懂的文案（阶段 4 / F3：文案归界面层）。

    【为什么在 app.py 而不在 qa.py】
      qa.py 的契约是"异常不吞、不包装"，它只负责抛；
      把异常翻译成人话是"最终用户边界"的责任，归界面层。

    【最小实现】靠消息关键字识别，不引入新异常类型（契约 4.9）。
      脆弱性可接受：识别错也只是文案不精确，不影响功能。

    【阶段 5】本函数只做映射，不记日志；日志在调用处记（便于带出 user_msg）。
    """
    msg = str(exc)
    if "ZHIPU_API_KEY" in msg:
        return "请先配置 ZHIPU_API_KEY"
    if "429" in msg or "1302" in msg or "1305" in msg or "rate limit" in msg.lower():
        return "生成服务繁忙，请稍后重试"
    if "chroma" in msg.lower() or "collection" in msg.lower():
        return "检索服务暂时不可用"
    return "服务暂时不可用，请稍后重试"


def _on_upload(file):
    """上传并建索引的回调。"""
    if not file:
        return "请先选择一个 PDF 文件。", gr.update(), gr.update(), gr.update()

    try:
        chunks = parse_and_chunk(file)
        build_index(chunks)
        doc_name = chunks[0].doc_name if chunks else "未知文档"
        status = f"已索引 {len(chunks)} 个 chunk，来自 {doc_name}。对话已重置。"
    except Exception as e:
        return f"索引失败：{e}", gr.update(), gr.update(), gr.update()

    return status, [], [], ""


def _on_send(query, history):
    """提问回调（流式版 + 阶段 4 降级/容错）。

    【流式约定（契约 3.4 / S2 / T5）】
      从 return 改为 yield：每收到一个 token 事件，yield 更新后的全量 history。
      Gradio 5.49.1 的 Chatbot 流式不支持「追加」，必须每次 yield 全量快照。
      引用区（citations_md）只在收到 citations 事件时更新，token 阶段用 gr.update()。

    【占位提示（阶段 3 / S4）】
      重排耗时约 9 秒，首字延迟全卡在这。点发送后立即 yield 一条「正在检索并重排…」。

    【pending_prefix（阶段 4 / G6 实现约束）】
      收到 degraded 时不立刻 append 空消息（会闪一次空条目），而是记 pending_prefix；
      第一个 token 到达时拼到 content 开头。若零 token 就异常，则流结束时拼前缀 + 提示。

    【异常处理（阶段 4 / 4.5）】
      永远 append 到现有 content，不重置——覆盖 D1+D2、D1+D3 叠加场景：
        - D1+D2：content 已有 degraded 前缀 → 追加"（生成服务繁忙…）"
        - D1+D3：content 已有前缀 + 半截答案 → 追加"（生成中断：原因）"
        - 纯 D2：content 还是占位 → 替换为"（生成服务繁忙…）"

    【未知事件类型（阶段 4 / G7）】
      print 到 stderr，不崩，向前兼容后续阶段新增的事件类型。
      保留 print 而非 logger：这是开发期告警，不属业务日志（契约 5e）。

    【阶段 5 / L2】
      异常时记一条 logger.error（结构化：error_type / error_msg / user_msg）。
      业务日志（query done）在 qa.py 里打，本函数不重复打。

    【性能隐患】
      每个 token 都 yield 全量 history，100 个 token ≈ 100 次序列化。
      Gradio 内部会 diff，实际开销可接受；若未来 token 很碎，可考虑按时间/字数节流。
    """
    if history is None:
        history = []

    if not query or not query.strip():
        new_history = history + [
            {"role": "assistant", "content": "请输入问题后再发送。"}
        ]
        yield new_history, new_history, "", EMPTY_CITATIONS
        return

    new_history = history + [{"role": "user", "content": query}]
    # 【核心逻辑】assistant 消息用同一个 dict 引用，每次 += 后 yield 全量快照。
    assistant_msg = {"role": "assistant", "content": ""}
    new_history = new_history + [assistant_msg]

    # 【阶段 3 / S4】占位提示：立即 yield，解决重排 9 秒的白屏焦虑。
    assistant_msg["content"] = PLACEHOLDER
    yield new_history, new_history, "", gr.update()

    pending_prefix = ""     # 阶段 4：degraded 前缀，等首 token 到达时拼接
    first_token = True      # 是否还没收到真 token（区分占位）
    citations_md = EMPTY_CITATIONS

    try:
        for event in answer_stream(query):
            etype = event["type"]
            if etype == "degraded":
                # D1：记前缀，不即时渲染（G6：避免空消息闪烁）。
                pending_prefix = DEGRADED_PREFIX
            elif etype == "token":
                if first_token:
                    # 首 token：用 pending_prefix + token 覆盖占位。
                    assistant_msg["content"] = pending_prefix + event["content"]
                    pending_prefix = ""
                    first_token = False
                else:
                    assistant_msg["content"] += event["content"]
                yield new_history, new_history, "", gr.update()
            elif etype == "citations":
                if first_token:
                    # 边界：零 token 但有 citations（LLM 返回空），用前缀收尾。
                    assistant_msg["content"] = pending_prefix + "（模型未返回内容）"
                    pending_prefix = ""
                    first_token = False
                citations_md = _format_citations(event["citations"])
                yield new_history, new_history, "", citations_md
            elif etype == "refused":
                # 拒答：覆盖占位（清空前缀，因为拒答不需要降级前缀）。
                assistant_msg["content"] = REFUSAL_TEXT
                pending_prefix = ""
                first_token = False
                yield new_history, new_history, "", EMPTY_CITATIONS
                return
            else:
                # G7：未知事件类型忽略并告警，不崩（向前兼容）。
                # 契约 5e：保留 print(stderr)，不改 loguru（非业务日志）。
                print(f"[警告] 未知事件类型: {etype}", file=sys.stderr)
    except Exception as e:
        # 阶段 4 / 4.5：异常 → 友好文案，永远 append 不重置。
        reason = _error_to_message(e)
        # 阶段 5 / L2：异常日志（结构化），带出 error_type / error_msg / user_msg。
        logger.bind(
            event="error",
            error_type=type(e).__name__,
            error_msg=str(e)[:200],
            user_msg=reason,
        ).error("query error")
        if first_token:
            # D2 或 D1+D2：一个真 token 都没出，替换占位（保留 degraded 前缀）。
            assistant_msg["content"] = pending_prefix + f"（{reason}）"
            citations_md = EMPTY_CITATIONS
        else:
            # D3 或 D1+D3：有半截答案，append 中断提示（G5：引用区标注中断）。
            assistant_msg["content"] += f"\n\n（{reason}）"
            citations_md = INTERRUPTED_CITATIONS
        yield new_history, new_history, "", citations_md


def build_app() -> gr.Blocks:
    """构建 Gradio 界面并返回 Blocks 对象。"""
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{SUBTITLE}")

        with gr.Row():
            file_input = gr.File(
                label="上传论文 PDF",
                file_types=[".pdf"],
                file_count="single",
            )
            upload_btn = gr.Button("上传并建索引")

        status_md = gr.Markdown("尚未上传论文。")

        # Gradio 5.x 必须显式声明 type="messages" 才能正确渲染字典列表
        chatbot = gr.Chatbot(label="对话", type="messages")
        history_state = gr.State([])

        with gr.Row():
            query_box = gr.Textbox(
                label="你的问题",
                placeholder="例如：How much does LoRA reduce trainable parameters?",
                scale=4,
            )
            send_btn = gr.Button("发送", scale=1)

        citations_md = gr.Markdown(EMPTY_CITATIONS)

        upload_btn.click(
            fn=_on_upload,
            inputs=[file_input],
            outputs=[status_md, chatbot, history_state, citations_md],
        )

        # 【阶段 3】_on_send 现在是生成器（yield），Gradio 会自动按流式渲染。
        send_btn.click(
            fn=_on_send,
            inputs=[query_box, history_state],
            outputs=[chatbot, history_state, query_box, citations_md],
        )
        query_box.submit(
            fn=_on_send,
            inputs=[query_box, history_state],
            outputs=[chatbot, history_state, query_box, citations_md],
        )

    return demo


if __name__ == "__main__":
    # 【新增 / 性能优化】通过魔搭内网极速下载模型
    # 将 1.2GB 的模型下载成本转移到服务启动阶段（约 10-20 秒），
    # 避免用户第一次提问时苦等 120 秒。
    print("正在检查模型完整性并下载（利用魔搭内网，极速）...")
    try:
        from modelscope import snapshot_download
        snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5')
        snapshot_download('BAAI/bge-reranker-base', local_dir='./models/bge-reranker-base')
        print("模型文件已就绪。")
    except Exception as e:
        print(f"模型下载失败，程序可能无法正常启动：{e}")

    # 【修复】启动时强制清空云端可能损坏或残留的旧索引数据。
    print("正在清理并初始化预置论文索引...")
    try:
        clear_index()

        default_pdf = "data/sample.pdf"
        if os.path.exists(default_pdf):
            chunks = parse_and_chunk(default_pdf)
            build_index(chunks)
            print(f"预置论文索引完成，共 {len(chunks)} 个 chunk。")
        else:
            print("未找到 data/sample.pdf，跳过预置索引。")
    except Exception as e:
        print(f"启动时自动索引失败（不影响应用启动）：{e}")

    # 【新增 / 性能优化】预热重排模型
    # 【背景】云端免费 CPU 冷启动时，CrossEncoder 是懒加载的——只有第一次
    #        调用 rerank() 才会读 1.1GB 模型。实测用户首次提问需要等 ~63 秒，
    #        第二次起只要 ~20 秒。本段把模型加载成本从「用户提问时」转移到
    #        「服务启动时」，用户第一次提问就只需等约 20 秒（检索 + LLM）。
    # 【为什么用 try/except】预热失败（例如模型文件缺失）不应阻止应用启动；
    #        用户提问时 rerank() 内部仍会重新触发加载，只是会慢一点。
    # 【为什么局部 import】避免 build_app 模块被外部 import 时连带加载 rerank，
    #        保持模块职责单一（界面模块不依赖重排模块，除非真的要启动服务）。
    print("正在预热重排模型（约需 30 秒，用户无感）...")
    try:
        from rerank import warmup as _warmup_rerank
        _warmup_rerank()
        print("重排模型预热完成。")
    except Exception as e:
        print(f"重排模型预热失败（不影响应用启动，提问时会尝试重新加载）：{e}")

    demo = build_app()
    # Gradio 5.x 的 SSE 机制本身不会超时，无需设置 timeout
    demo.queue(default_concurrency_limit=1)

    if os.environ.get("MODELSCOPE_ENVIRONMENT"):
        print("检测到魔搭云端环境，启动于 0.0.0.0:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("检测到本地环境，使用默认端口启动。")
        demo.launch()