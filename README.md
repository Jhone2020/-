# 📚 RAG 知识库问答系统

基于 LangChain + DeepSeek + ChromaDB 的企业级知识库问答系统，支持多种文档格式上传、智能检索、对话历史管理等功能。

## ✨ 功能特性

- 📄 **多格式文档支持** - 支持 PDF、TXT、DOCX 格式文档上传
- 🔍 **智能检索增强** - 基于向量数据库的语义检索，精准匹配相关内容
- 💬 **对话式问答** - 自然语言交互，基于文档内容回答问题
- 📊 **批量上传** - 支持批量上传多个文档，高效构建知识库
- 🔗 **来源追溯** - 自动标注回答引用的文档来源
- 🎨 **关键词高亮** - 自定义关键词高亮显示，快速定位关键信息
- 💾 **对话历史** - 自动保存对话记录，支持历史会话加载
- 🚀 **LangChain 原生** - 完整的 LangChain 集成，支持 LangSmith 追踪

## 🛠️ 技术栈

- **前端框架**: Streamlit
- **AI 模型**: DeepSeek API (deepseek-chat)
- **向量数据库**: ChromaDB
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2 (80MB)
- **文档解析**: PyPDF2, python-docx
- **框架**: LangChain

## 📦 安装

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <project-directory>
2. 安装依赖
bash
pip install -r requirements.txt
requirements.txt 内容：

txt
streamlit
python-dotenv
langchain-deepseek
langchain-core
langchain-community
langchain-huggingface
chromadb
huggingface-hub
sentence-transformers
PyPDF2
python-docx
3. 配置环境变量
创建 .env 文件并配置 DeepSeek API Key：

env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
获取 DeepSeek API Key：访问 DeepSeek 官网 注册并获取 API Key

4. 配置 HuggingFace 镜像（可选）
代码已配置国内镜像加速：

python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
🚀 运行
bash
streamlit run app.py