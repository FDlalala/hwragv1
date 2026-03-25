"""
app.py —— 基于 Gradio 的 RAG Agent 可视化交互界面
直接复用 rag_agent_local.py 中的 agent、retrieve_context 工具和向量库。

启动方式：
    python app.py
    # 或指定端口
    python app.py --port 7860
"""

import argparse
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

# ========== 导入 Agent 核心（复用 rag_agent_local.py）==========
from rag_agent_local import agent, vectordb, RETRIEVE_K, _last_retrieved_docs as _docs_ref
import rag_agent_local as rag_module

# ========== 流式问答核心函数 ==========
def chat_stream(user_message: str, history: list):
    """
    流式生成回答，同时收集检索来源。
    history 格式：[{"role": "user"/"assistant", "content": "..."}]
    """
    if not user_message.strip():
        return

    # 重置检索文档缓存
    rag_module._last_retrieved_docs = []

    # 构建消息历史（多轮对话）
    # Gradio 6.x history 格式: [{"role": "user"/"assistant", "content": "..."}]
    messages = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "user" and content:
            messages.append(HumanMessage(content=content))
        elif role == "assistant" and content and "⏳" not in content:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    # 流式调用 Agent
    partial_answer = ""
    tool_called = False

    for event in agent.stream(
        {"messages": messages},
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        msg_type = last_msg.__class__.__name__

        if msg_type == "AIMessage":
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_called = True
                # 工具调用阶段：显示"检索中..."提示
                yield "⏳ 正在检索知识库...", []
            elif last_msg.content:
                # 最终回答：逐字流式输出
                partial_answer = last_msg.content
                # 收集来源（此时检索已完成）
                sources = _build_sources(rag_module._last_retrieved_docs)
                yield partial_answer, sources

    # 如果没有触发工具，也要返回最终结果
    if not tool_called and partial_answer == "":
        # 兜底：非流式情况
        result = agent.invoke({"messages": messages})
        final = result["messages"][-1].content
        sources = _build_sources(rag_module._last_retrieved_docs)
        yield final, sources


def _build_sources(docs: list) -> list:
    """将检索文档列表转换为来源信息列表"""
    sources = []
    for i, doc in enumerate(docs, 1):
        case_name = doc.metadata.get("case_name", doc.metadata.get("title", "未知案例"))
        case_id   = doc.metadata.get("case_id",   doc.metadata.get("source", "未知ID"))
        snippet   = doc.page_content[:120].replace("\n", " ").strip()
        sources.append({
            "index":     i,
            "case_name": case_name,
            "case_id":   case_id,
            "snippet":   snippet,
        })
    return sources


def format_sources_html(sources: list) -> str:
    """将来源列表渲染为 HTML 展示"""
    if not sources:
        return "<div style='color:#888; padding:8px;'>本次未检索知识库</div>"

    html = ""
    for s in sources:
        html += f"""
        <div style='
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            background: #f9f9f9;
        '>
            <div style='font-weight:bold; color:#1a73e8; margin-bottom:4px;'>
                [{s['index']}] {s['case_name']}
            </div>
            <div style='font-size:12px; color:#666; margin-bottom:4px;'>
                案例ID：{s['case_id']}
            </div>
            <div style='font-size:13px; color:#444; line-height:1.5;'>
                {s['snippet']}…
            </div>
        </div>
        """
    return html


# ========== Gradio 界面 ==========
def build_ui():
    doc_count = vectordb._collection.count()

    with gr.Blocks(
        title="案例知识库 RAG Agent",
        theme=gr.themes.Soft(),
        css="""
        #chatbot { height: 520px; }
        #sources-panel { height: 520px; overflow-y: auto; }
        .status-bar { font-size: 13px; color: #555; padding: 4px 0; }
        """
    ) as demo:

        # ── 标题 ──
        gr.Markdown(
            "# 🤖 案例知识库 RAG Agent\n"
            "基于 Agentic RAG，自动判断是否需要检索知识库，支持多轮对话。"
        )

        with gr.Row():
            # ── 左侧：对话区 ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="对话",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请输入问题，例如：有没有关于XX的案例？",
                        label="",
                        scale=5,
                        lines=1,
                        autofocus=True,
                    )
                    send_btn = gr.Button("发送 ▶", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

            # ── 右侧：状态 + 来源 ──
            with gr.Column(scale=2):
                gr.Markdown("### 📊 系统状态")
                gr.HTML(f"""
                <div class='status-bar'>
                    📚 向量库案例数：<b>{doc_count}</b> 条<br>
                    🔍 检索 top-k：<b>{RETRIEVE_K}</b><br>
                    🤖 模型：<b>qwen3</b>
                </div>
                """)

                gr.Markdown("### 🔍 本次检索来源")
                sources_display = gr.HTML(
                    value="<div style='color:#888; padding:8px;'>暂无检索记录</div>",
                    elem_id="sources-panel",
                    label="",
                )

        # ── 状态存储 ──
        sources_state = gr.State([])

        # ── 事件：发送消息 ──
        def user_submit(user_msg, history):
            """用户提交后立即把用户消息加入历史，清空输入框"""
            history = history or []
            history.append({"role": "user", "content": user_msg})
            return "", history

        def bot_respond(history, sources):
            """流式生成 bot 回答，同步更新来源面板"""
            if not history:
                yield history, sources, format_sources_html(sources)
                return

            user_msg = history[-1]["content"]
            # 去掉最后一条（用户消息），传入历史
            prev_history = history[:-1]

            # 先 yield 占位消息，避免空白
            yield history + [{"role": "assistant", "content": "⏳ 思考中..."}], sources, format_sources_html(sources)

            for partial_text, new_sources in chat_stream(user_msg, prev_history):
                if new_sources:
                    sources = new_sources
                yield history + [{"role": "assistant", "content": partial_text}], sources, format_sources_html(sources)

        def clear_all():
            return [], [], "<div style='color:#888; padding:8px;'>暂无检索记录</div>"


        # 绑定事件
        msg_input.submit(
            user_submit,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False,
        ).then(
            bot_respond,
            inputs=[chatbot, sources_state],
            outputs=[chatbot, sources_state, sources_display],
        )

        send_btn.click(
            user_submit,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False,
        ).then(
            bot_respond,
            inputs=[chatbot, sources_state],
            outputs=[chatbot, sources_state, sources_display],
        )

        clear_btn.click(
            clear_all,
            outputs=[chatbot, sources_state, sources_display],
        )

    return demo


# ========== 启动 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int,  default=7860,  help="监听端口")
    parser.add_argument("--host",   type=str,  default="0.0.0.0", help="监听地址")
    parser.add_argument("--share",  action="store_true",      help="生成公网分享链接")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue()   # 开启队列，支持流式输出
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
