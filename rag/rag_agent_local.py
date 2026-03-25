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
    persist_directory="./chroma_cases"
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
        title  = doc.metadata.get("case_name", doc.metadata.get("title", "未知标题"))
        source = doc.metadata.get("case_id",   doc.metadata.get("source", "未知来源"))
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        print(f"  [{i}] 案例名: {title}")
        print(f"       案例ID: {source}")
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
    "你是一个专业的AI助手，拥有一个案例知识库检索工具 retrieve_context。\n"
    "\n"
    "【判断是否需要检索】\n"
    "- 如果用户是闲聊、打招呼、问你是谁、问天气等与知识库无关的问题，直接用中文友好回答，不要调用工具。\n"
    "- 如果用户提出了需要查询案例、事实、专业知识的问题，必须先调用 retrieve_context 工具检索，再基于检索结果回答。\n"
    "\n"
    "【检索后的回答规则】\n"
    "- 如果检索结果与问题相关，请用中文简洁回答，并说明参考了哪些案例。\n"
    "- 如果检索结果与问题完全无关，请回答'知识库中没有找到相关信息'，不要编造答案。\n"
    "- 不要使用知识库外的常识补充，不要猜测。\n"
    "\n"
    "示例：\n"
    "  用户: 你好 → 直接回答，不检索\n"
    "  用户: 有没有关于XX的案例 → 调用 retrieve_context → 基于结果回答"
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
if __name__ == "__main__":
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
                title  = doc.metadata.get("case_name", doc.metadata.get("title",  "未知标题"))
                source = doc.metadata.get("case_id",   doc.metadata.get("source", "未知来源"))
                print(f"  [{i}] 案例名: {title}")
                print(f"       案例ID: {source}")
            print("-"*50)
        else:
            print("\n（本次未检索知识库，无参考来源）")
