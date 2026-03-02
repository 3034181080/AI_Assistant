import streamlit as st
from openai import OpenAI
import pdfplumber
import os
import tempfile
import uuid

# 引入 RAG 相关库
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 智能获取 API Key
try:
    deepseek_key = st.secrets["DEEPSEEK_API_KEY"]
    dashscope_key = st.secrets["DASHSCOPE_API_KEY"]
except (KeyError, FileNotFoundError):
    from dotenv import load_dotenv
    load_dotenv()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")

if not deepseek_key or not dashscope_key:
    st.error("致命错误：找不到 API Key！请检查 Secrets 或 .env 配置！")
    st.stop()

# 设置环境变量，�� LangChain 底层调用
os.environ["DASHSCOPE_API_KEY"] = dashscope_key

client = OpenAI(
    api_key=deepseek_key, 
    base_url="https://api.deepseek.com"
)

st.set_page_config(page_title="企业知识库 AI (RAG 版)", page_icon="📄")
st.title("智能私有知识库 (支持超大文档)")

# 初始化 Session State
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "老板好，我是你的私有数据助手。请在左侧上传文档，等待数据库建立完成后再向我提问！"}]

# ================= 侧边栏：文档处理与建库 =================
with st.sidebar:
    st.header("1. 上传私有数据")
    uploaded_file = st.file_uploader("支持 PDF/TXT (可处理长文档)", type=["pdf", "txt"])
    
    if uploaded_file is not None and st.session_state.vector_db is None:
        with st.spinner("正在解析文档并建立专属知识库，请稍候..."):
            doc_text = ""
            if uploaded_file.name.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text(layout=True) 
                        if text:
                            doc_text += text + "\n"
            else:
                doc_text = uploaded_file.getvalue().decode("utf-8")
            
            # 1. 文本切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 网页版切块可以大一点
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(doc_text)
            
            # 2. 建立向量数据库 (存入临时文件夹，保证每次会话独立)
            temp_dir = tempfile.mkdtemp()
            db_path = os.path.join(temp_dir, f"chroma_{uuid.uuid4().hex[:8]}")
            
            embedding_model = DashScopeEmbeddings(model="text-embedding-v2")
            st.session_state.vector_db = Chroma.from_texts(
                texts=chunks, 
                embedding=embedding_model, 
                persist_directory=db_path
            )
            
            st.success(f"知识库建立完成！共分为 {len(chunks)} 个数据块。")

# ================= 聊天界面 =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 核心逻辑：如果建好了数据库，就去检索
        context_text = ""
        if st.session_state.vector_db is not None:
            # 检索最相关的 3 个文档块
            retrieved_docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            for i, doc in enumerate(retrieved_docs):
                context_text += f"[片段 {i+1}]: {doc.page_content}\n"
            
            # 如果想在网页上看看它捞出了什么，可以取消下面这行的注释
            # st.info(f"后台检索到的相关片段：\n{context_text}")

        # 构造发给大模型的提示词
        if context_text:
            system_prompt = f"你是一个专业的数据分析师。请严格根据以下背景资料回答问题。\n背景资料：\n{context_text}"
        else:
            system_prompt = "你是一个智能助手，请正常回答用户的问题。"

        # 调用大模型流式输出
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            message_placeholder.markdown(f"服务器开小差了，请稍后再试。错误信息：{str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})