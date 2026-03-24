import os
import warnings
warnings.filterwarnings("ignore")

from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ========== 1. Embeddings ==========
print("加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="./bge-small-zh-v1.5"
)

# ========== 2. 向量数据库 ==========
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_web"
)
print(f"向量库文档数量: {vectordb._collection.count()}")

# ========== 3. 检索工具 ==========
# 可调参数：k 控制返回文档数，建议范围 2~6
RETRIEVE_K = 5

# 全局变量，用于在最终回答时展示引用来源
_last_retrieved_docs = []

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    global _last_retrieved_docs
    # ---------- 醒目标记：工具被触发 ----------
    print("\n" + "★"*50)
    print("★★★  【检索工具被调用！】  ★★★")
    print("★"*50)
    # ---------- 可观察日志：开始 ----------
    print(f"[检索] 原始 query: {query}")
    print(f"[检索] 检索 top-k = {RETRIEVE_K}")
    # ---------- 执行检索 ----------
    retrieved_docs = vectordb.similarity_search(query, k=RETRIEVE_K)
    _last_retrieved_docs = retrieved_docs
    # ---------- 打印每条结果摘要 ----------
    for i, doc in enumerate(retrieved_docs, 1):
        title  = doc.metadata.get("title",  "未知标题")
        source = doc.metadata.get("source", "未知来源")
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        print(f"  [{i}] 标题: {title}")
        print(f"       来源: {source}")
        print(f"       摘要: {snippet}...")
    print("★"*50 + "\n")
    # ---------- 拼接返回字符串 ----------
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# ========== 4. API LLM ==========
llm = ChatOpenAI(
    model="qwen3",
    openai_api_base="url",  # 替换为你的完整 URL
    openai_api_key="EMPTY",          # 本地服务无需 key，填 EMPTY 即可
    temperature=0,
    max_tokens=512,
)
print("LLM API 加载完成！")

# ========== 5. Agent ==========
SYSTEM_PROMPT = (
    "你是一个专业的AI助手，拥有一个知识库检索工具 retrieve_context。\n"
    "【强制规则】对于任何用户问题，你的第一步操作必须是调用 retrieve_context 工具进行检索，"
    "禁止在调用工具之前直接回答问题。\n"
    "调用工具后，根据检索结果组织回答。如果检索内容中没有足够信息，"
    "请明确回答'知识库中没有找到相关信息'，"
    "不要使用知识库外的常识补充，不要猜测，不要编造答案。\n"
    "回答时优先依据检索结果，请用中文回答。\n"
    "示例流程：用户提问 → 调用 retrieve_context(query=用户问题) → 阅读检索结果 → 给出回答。"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

agent = create_react_agent(
    model=llm,
    tools=[retrieve_context],
    prompt=prompt
)

# ========== 6. 交互问答 ==========
print("\n" + "="*50)
print("RAG Agent 已就绪，输入 quit 退出")
print("="*50)

while True:
    query = input("\n请输入问题: ").strip()
    if query.lower() in ["quit", "exit", "q"]:
        break
    if not query:
        continue

    _last_retrieved_docs = []  # 每次提问前清空
    print("\n" + "-"*50)

    final_answer = None
    tool_called = False

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        msg_type = last_msg.__class__.__name__

        # 打印 AI 消息（包括工具调用请求 和 最终回答）
        if msg_type == "AIMessage":
            # 如果有 tool_calls，说明是工具调用请求
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tool_called = True
                print("\n[Agent 决策] 准备调用工具:")
                for tc in last_msg.tool_calls:
                    print(f"  工具名: {tc['name']}")
                    print(f"  参数:   {tc['args']}")
            else:
                # 最终回答
                final_answer = last_msg.content
                print("\n" + "="*50)
                print("【最终回答】")
                print("="*50)
                print(final_answer)

        # 打印工具返回结果（ToolMessage）
        elif msg_type == "ToolMessage":
            print("\n[工具返回] 检索完成，内容已传递给 LLM（见上方检索日志）")

    # 如果工具从未被调用，给出警告
    if not tool_called:
        print("\n⚠️  警告：本次回答未触发检索工具，回答可能基于模型自身知识而非知识库！")

    # 打印引用来源汇总
    if _last_retrieved_docs:
        print("\n" + "-"*50)
        print("【本次回答参考的知识库来源】")
        print("-"*50)
        for i, doc in enumerate(_last_retrieved_docs, 1):
            title  = doc.metadata.get("title",  "未知标题")
            source = doc.metadata.get("source", "未知来源")
            print(f"  [{i}] 标题: {title}")
            print(f"       来源: {source}")
        print("-"*50)
    else:
        print("\n（本次未检索知识库，无参考来源）")
