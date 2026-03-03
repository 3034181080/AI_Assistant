import streamlit as st
from openai import OpenAI
import pdfplumber
import os
import tempfile
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 1. 智能获取 API Key
try:
    deepseek_key = st.secrets["DEEPSEEK_API_KEY"]
    dashscope_key = st.secrets["DASHSCOPE_API_KEY"]
except (KeyError, FileNotFoundError):
    from dotenv import load_dotenv
    load_dotenv()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")

if not deepseek_key or not dashscope_key:
    st.error("系统配置错误：缺少必要的 API 密钥。")
    st.stop()

# 2. 设置底层环境变量
os.environ["DASHSCOPE_API_KEY"] = dashscope_key

client = OpenAI(
    api_key=deepseek_key, 
    base_url="https://api.deepseek.com"
)

# 3. 页面基础配置
st.set_page_config(page_title="企业知识库 AI (RAG 版)", layout="wide")
st.title("智能私有知识库系统")
st.markdown("基于 RAG 架构与多模型协同，支持超大容量文档的语义级检索与问答。")

# 4. 初始化与重置 Session State
def reset_session():
    st.session_state.vector_db = None
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是您的私有数据分析终端。请先在左侧上传文档，等待知识库建立完成后再进行提问。"}]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是您的私有数据分析终端。请先在左侧上传文档，等待知识库建立完成后再进行提问。"}]

# ================= 5. 侧边栏：文档处理与建库模块 =================
with st.sidebar:
    st.header("数据接入")
    uploaded_file = st.file_uploader("支持 PDF/TXT (可处理长文档)", type=["pdf", "txt"])
    
    if uploaded_file is not None and st.session_state.vector_db is None:
        with st.spinner("正在解析文档并构建向量知识库，请稍候..."):
            doc_text = ""
            try:
                if uploaded_file.name.endswith(".pdf"):
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text(layout=True) 
                            if text:
                                doc_text += text + "\n"
                else:
                    doc_text = uploaded_file.getvalue().decode("utf-8")
                
                if not doc_text.strip():
                    st.error("解析失败：未能提取到有效文本，请检查文件是否为纯图片扫描件。")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_text(doc_text)
                    
                    temp_dir = tempfile.mkdtemp()
                    db_path = os.path.join(temp_dir, f"chroma_{uuid.uuid4().hex[:8]}")
                    
                    embedding_model = DashScopeEmbeddings(model="text-embedding-v2")
                    st.session_state.vector_db = Chroma.from_texts(
                        texts=chunks, 
                        embedding=embedding_model, 
                        persist_directory=db_path
                    )
                    
                    st.success(f"知识库构建完成。共生成 {len(chunks)} 个数据块，已就绪。")
            except Exception as e:
                st.error(f"处理文档时发生异常：{str(e)}")

    st.divider()
    
    if st.button("清空知识库与对话", use_container_width=True, type="primary"):
        reset_session()
        st.rerun()

# ================= 6. 主聊天界面模块 =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    if st.session_state.vector_db is None:
        st.warning("请求被拦截：请先在左侧上传文档并等待知识库建立完成。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            context_text = ""
            try:
                retrieved_docs = st.session_state.vector_db.similarity_search(prompt, k=3)
                for i, doc in enumerate(retrieved_docs):
                    context_text += f"[参考片段 {i+1}]: {doc.page_content}\n"
            except Exception as e:
                st.error(f"检索数据库时发生错误：{str(e)}")
                st.stop()

            system_prompt = f"你是一个专业的数据分析师。请严格基于以下背景资料回答问题，如果没有相关信息，请回答“未找到相关数据”。\n背景资料：\n{context_text}"

            api_messages = [{"role": "system", "content": system_prompt}]
            
            history = st.session_state.messages[1:-1][-4:] 
            for msg in history:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
                
            api_messages.append({"role": "user", "content": prompt})

            try:
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=api_messages,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                message_placeholder.markdown(f"服务异常，请稍后重试。详细信息：{str(e)}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})