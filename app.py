import streamlit as st
from openai import OpenAI
import pdfplumber
import os
from dotenv import load_dotenv

# 加载 .env 文件里的密码
load_dotenv()

# 从环境变量中安全地获取 API Key
# 这样哪怕代码开源，别人也看不到你的 Key
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"
)

# ... 下面原本的代码完全保持不变 ...

st.set_page_config(page_title="企业知识库 AI", page_icon="📄")
st.title("📄 企业级私有知识库助手")

# ================= 核心商业逻辑：侧边栏上传文档 =================
with st.sidebar:
    st.header("1. 喂给 AI 私有数据")
    uploaded_file = st.file_uploader("请上传一份 PDF 或 TXT 文档", type=["pdf", "txt"])
    
    doc_text = ""
    if uploaded_file is not None:
        # 判断文件类型并提取文字
        if uploaded_file.name.endswith(".pdf"):
    # 使用更强大的 pdfplumber 解析中文和表格
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
            # 提取文本，并尽量保留表格的排版格式
                    text = page.extract_text(layout=True) 
            if text:
                doc_text += text + "\n"
            
        st.success(f"✅ 文件解析成功！共提取了 {len(doc_text)} 个字符。")
        # 增加一个开关，让用户决定要不要看提取出来的纯文本
        with st.expander("查看提取的文本片段"):
            st.write(doc_text[:500] + "...")

# ================= 聊天界面逻辑 =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "老板好，请在左侧上传文档，然后向我提问！"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入基于文档的问题..."):
    # 1. 记录用户输入并在网页显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 构造传给 AI 的完整消息（重点！！）
    # 如果上传了文档，我们就动态构造一个 "系统提示词(System Prompt)"，强迫大模型基于文档回答
    api_messages = []
    if doc_text:
        system_prompt = f"""
        你是一个专业的企业数据分析师。请你严格基于以下【参考文档】的内容来回答用户的问题。
        如果用户的提问在文档中找不到答案，请直接回答“抱歉，文档中未提及相关信息”，绝对不要自己瞎编。
        
        【参考文档】：
        {doc_text[:8000]} # 限制一下长度，防止一次性传入过多字符导致 API 报错
        """
        api_messages.append({"role": "system", "content": system_prompt})
    
    # 把历史对话拼接在系统提示词后面
    api_messages.extend(st.session_state.messages)

    # 3. 调用 API 生成回复
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="deepseek-chat",  
            messages=api_messages,  # 注意这里传的是我们拼接好的 api_messages
            stream=True 
        )
        response = st.write_stream(stream)
        
    st.session_state.messages.append({"role": "assistant", "content": response})