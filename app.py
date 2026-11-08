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
  - rerank ：启动预热时使用（局部 import，避免模块被 import 时连带加载）。
"""
import os
import gradio as gr

from ingest import parse_and_chunk
from index_store import build_index, clear_index
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