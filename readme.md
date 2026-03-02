# 企业级私有知识库 AI 助手 (RAG System)

## 项目简介
本项目是一个基于大语言模型（LLM）的轻量级本地知识库检索系统。利用 Streamlit 构建极速前端界面，结合 `pdfplumber` 解决复杂的中文与表格 PDF 解析痛点，实现了将企业本地私有数据“喂”给 AI，打造专属智能客服与数据分析助手。

## 技术栈
- **前端交互:** Streamlit (支持纯 Python 极速构建现代 UI)
- **大模型驱动:** DeepSeek API (完全兼容 OpenAI 接口规范)
- **文档解析:** pdfplumber (针对中文与表格高精度提取)
- **核心逻辑:** Prompt Injection, 上下文会话记忆机制

## 核心特性
1. **多格式文档解析**：支持 TXT 与复杂排版的 PDF 文档读取。
2. **私有化数据隔离**：AI 回答严格基于用户上传的文档，拒绝“幻觉”和胡编乱造。
3. **流式输出体验**：采用 Stream 打字机效果，响应无延迟。
4. **上下文连续对话**：基于 session_state 实现持久化记忆功能。

## 极速启动指南
```bash
# 1. 克隆仓库
git clone https://github.com/3034181080/AI-Knowledge-Base.git

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行应用
streamlit run app.py
```

## 👨‍💻 开发者
- **Author:** [DDDD]
