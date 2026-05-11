"""
Web RAG System - 基于 Streamlit 的文档问答系统
使用 bge-m3 作为 embedding 模型，支持 PDF 文档上传和问答
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import os
import pickle
import requests
import urllib3
import time
import hashlib

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'

from dotenv import load_dotenv
load_dotenv()

# 配置
API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')
LOCAL_MODEL_PATH = "./models/Xorbits/bge-m3"
VECTOR_DB_PATH = "./vector_db"

# 页面配置
st.set_page_config(
    page_title="RAG 文档问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-container {
        height: calc(100vh - 100px);
    }
    .chat-container {
        height: calc(100vh - 250px);
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        margin-right: 20%;
    }
    .source-info {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #fafafa;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def call_llm(prompt: str, model: str = "glm-5.1") -> str:
    """调用 LLM API"""
    url = BASE_URL.rstrip('/') + '/chat/completions'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}]
    }

    response = requests.post(url, headers=headers, json=data, timeout=60, verify=False)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']


@st.cache_resource
def get_embeddings():
    """获取嵌入模型（缓存）"""
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        model_path = LOCAL_MODEL_PATH
    else:
        from modelscope import snapshot_download
        model_path = snapshot_download(
            'Xorbits/bge-m3',
            cache_dir='./models'
        )

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """从PDF中提取文本并记录页码"""
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))

    return text, page_numbers


def process_text_with_splitter(text: str, page_numbers: List[int]) -> FAISS:
    """处理文本并创建向量存储"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    embeddings = get_embeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    # 记录页码信息
    lines = text.split("\n")
    page_info = {}
    for chunk in chunks:
        start_idx = text.find(chunk[:100])
        if start_idx == -1:
            for i, line in enumerate(lines):
                if chunk.startswith(line[:min(50, len(line))]):
                    start_idx = i
                    break
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if line and line in chunk:
                        start_idx = text.find(line)
                        break

        if start_idx != -1:
            line_count = text[:start_idx].count("\n")
            if line_count < len(page_numbers):
                page_info[chunk] = page_numbers[line_count]
            else:
                page_info[chunk] = page_numbers[-1] if page_numbers else 1
        else:
            page_info[chunk] = -1

    knowledgeBase.page_info = page_info
    return knowledgeBase


def get_file_hash(uploaded_file) -> str:
    """计算文件哈希值"""
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(content).hexdigest()


def process_pdf(uploaded_file) -> FAISS:
    """处理上传的 PDF 文件"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("📄 正在读取 PDF 文件...")
    progress_bar.progress(20)

    pdf_reader = PdfReader(uploaded_file)
    text, page_numbers = extract_text_with_page_numbers(pdf_reader)

    status_text.text(f"📝 提取了 {len(text)} 个字符，正在分割文本...")
    progress_bar.progress(40)

    knowledgeBase = process_text_with_splitter(text, page_numbers)

    status_text.text("✅ 文档处理完成！")
    progress_bar.progress(100)

    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    return knowledgeBase


def main():
    st.title("📚 RAG 文档问答系统")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None

    # 布局：左侧上传，右侧聊天
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📁 文档管理")

        # 文档上传区域
        uploaded_file = st.file_uploader(
            "上传 PDF 文档",
            type=["pdf"],
            help="支持 PDF 格式的文档"
        )

        if uploaded_file is not None:
            file_hash = get_file_hash(uploaded_file)

            # 检查是否是新文件
            if st.session_state.current_doc != file_hash:
                st.session_state.current_doc = file_hash
                st.session_state.knowledge_base = process_pdf(uploaded_file)
                st.session_state.messages = []  # 清空历史消息
                st.success(f"✅ 已加载: {uploaded_file.name}")
            else:
                st.info(f"📄 当前文档: {uploaded_file.name}")

        # 显示文档状态
        if st.session_state.knowledge_base is not None:
            st.divider()
            st.write("**文档状态**")
            st.write("✅ 向量数据库已就绪")
            if st.button("🗑️ 清除文档", type="secondary"):
                st.session_state.knowledge_base = None
                st.session_state.current_doc = None
                st.session_state.messages = []
                st.rerun()

        # 使用说明
        st.divider()
        st.write("**使用说明**")
        st.markdown("""
        1. 上传 PDF 文档
        2. 等待文档处理完成
        3. 在右侧输入问题
        4. 获取基于文档的回答
        """)

    with col_right:
        st.subheader("💬 对话")

        # 聊天历史显示区域
        chat_container = st.container()

        with chat_container:
            # 显示历史消息
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message:
                        st.markdown(f"<div class='source-info'>📖 来源: {message['sources']}</div>",
                                   unsafe_allow_html=True)

        # 输入区域
        if st.session_state.knowledge_base is not None:
            if prompt := st.chat_input("输入您的问题..."):
                # 显示用户消息
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                # 搜索相关文档
                with st.spinner("🔍 正在搜索相关内容..."):
                    docs = st.session_state.knowledge_base.similarity_search(prompt, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # 获取来源页码
                    sources = []
                    unique_pages = set()
                    for doc in docs:
                        text_content = getattr(doc, "page_content", "")
                        source_page = st.session_state.knowledge_base.page_info.get(
                            text_content.strip(), "未知"
                        )
                        if source_page not in unique_pages:
                            unique_pages.add(source_page)
                            sources.append(f"第{source_page}页")

                    # 构建 prompt 并调用 LLM
                    full_prompt = f"""根据以下上下文回答问题。如果上下文中没有相关信息，请说"根据文档内容，我无法回答这个问题"。

上下文:
{context}

问题: {prompt}

请用简洁、准确的语言回答:"""

                    response = call_llm(full_prompt)

                # 显示助手回复
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        if sources:
                            st.markdown(f"<div class='source-info'>📖 来源: {', '.join(sources)}</div>",
                                       unsafe_allow_html=True)

                # 保存助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": ', '.join(sources) if sources else None
                })

                st.rerun()
        else:
            st.info("👈 请先上传 PDF 文档开始对话")

            # 显示示例
            st.divider()
            st.write("**示例问题**")
            example_questions = [
                "文档的主要内容是什么？",
                "有哪些重要的规定？",
                "请总结文档的核心要点"
            ]
            for q in example_questions:
                st.markdown(f"- {q}")


if __name__ == "__main__":
    main()
