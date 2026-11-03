"""PaperLens 第 4/5 块：Gradio 界面 + 云端启动适配。

【模块职责】
  把前三块的「命令行能力」包成一个网页：
    上传 PDF -> 建索引 -> 提问 -> 显示答案和引用。
  同时适配本地开发与 ModelScope Docker 云端部署。

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
"""
import os
import gradio as gr

from ingest import parse_and_chunk
from index_store import build_index
from qa import answer

# ---------------------------------------------------------------------------
# 界面文案集中定义，避免散落在组件定义里。
# ---------------------------------------------------------------------------
TITLE = "PaperLens — 学术论文 RAG 助手"
SUBTITLE = "上传论文 PDF，基于原文回答问题，并给出页码级引用。"
EMPTY_CITATIONS = "（无引用）"


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
    """提问回调。"""
    if history is None:
        history = []

    if not query or not query.strip():
        new_history = history + [
            {"role": "assistant", "content": "请输入问题后再发送。"}
        ]
        return new_history, new_history, "", EMPTY_CITATIONS

    new_history = history + [{"role": "user", "content": query}]

    try:
        result = answer(query)
        new_history = new_history + [
            {"role": "assistant", "content": result.answer}
        ]
        citations_md = _format_citations(result.citations)
    except Exception as e:
        new_history = new_history + [
            {"role": "assistant", "content": f"出错了：{e}"}
        ]
        citations_md = EMPTY_CITATIONS

    return new_history, new_history, "", citations_md


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
    # 启动时自动索引 data/sample.pdf（应对云端非持久化磁盘）
    print("正在检查并初始化预置论文索引...")
    try:
        default_pdf = "data/sample.pdf"
        if os.path.exists(default_pdf):
            chunks = parse_and_chunk(default_pdf)
            build_index(chunks)
            print(f"预置论文索引完成，共 {len(chunks)} 个 chunk。")
        else:
            print("未找到 data/sample.pdf，跳过预置索引。")
    except Exception as e:
        print(f"启动时自动索引失败（不影响应用启动）：{e}")

    demo = build_app()
    # Gradio 5.x 的 SSE 机制本身不会超时，无需设置 timeout
    demo.queue(default_concurrency_limit=1)

    if os.environ.get("MODELSCOPE_ENVIRONMENT"):
        print("检测到魔搭云端环境，启动于 0.0.0.0:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("检测到本地环境，使用默认端口启动。")
        demo.launch()