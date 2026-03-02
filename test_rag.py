import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 引入阿里的在线词向量服务
from langchain_community.embeddings import DashScopeEmbeddings
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore")

# ================== 核心配置区 ==================
# 1. 填入你的大模型 Key (用于最后的对话回答)
deepseek_api_key = "sk-c2f789f8eccf45999726964f3b0df570"

# 2. 填入你的通义千问 Key (专门用于文本向量化，不用下载任何模型)
os.environ["DASHSCOPE_API_KEY"] = "sk-c7b51086d37345bbb4ce039236704bc3"
# ================================================

long_document_text = """
【公司机密档案：2026年星际开发计划】
第1条：本公司决定于2026年3月发射“朱德龙号”火星探测器。
第2条：该探测器的核心能源是一种叫“冰摇柠檬茶”的新型液体燃料。
第3条：探测器的指令长是一位名叫朱德龙的顶尖黑客，他精通 Python 和 AI RAG 技术。
第4条：火星基地的建设预算为 500 亿人民币，主要用于购买高端显卡。
第5条：如果遇到外星人，我们的谈判口号是：“Hello World, Show me your code.”
"""

print("正在切分文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
chunks = text_splitter.split_text(long_document_text)

print("正在调用云端 API 进行向量化并建立数据库...")
# 使用千问的在��模型进行向量化，无需下载本地模型文件
embedding_model = DashScopeEmbeddings(model="text-embedding-v2")

vector_db = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_db")

user_question = "探测器用的是什么燃料？"
print(f"\n用户提问：{user_question}")

retrieved_docs = vector_db.similarity_search(user_question, k=2)

print("\n检索到的相关片段：")
context_text = ""
for i, doc in enumerate(retrieved_docs):
    print(f"片段 {i+1}: {doc.page_content}")
    context_text += doc.page_content + "\n"

print("\n正在调用 DeepSeek 进行最终解答...")
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

system_prompt = f"""
你是一个专业的答疑助手。请你[只根据]以下提供的背景资料来回答问题。
如果资料里没有，请说“不知道”。
背景资料：
{context_text}
"""

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
)

print(f"\nAI 最终回答：\n{response.choices[0].message.content}")