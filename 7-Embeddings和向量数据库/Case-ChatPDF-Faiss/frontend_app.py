"""
Web RAG System 前端 - 简洁界面设计
"""

import streamlit as st
import requests
import time

# 页面配置
st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 后端 API 地址
API_BASE_URL = "http://localhost:8000"

# 自定义 CSS - 简洁风格
st.markdown("""
<style>
    /* 整体布局 */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 800px;
    }

    /* 标题 */
    .main-title {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 20px;
        margin-top: 30px;
        text-align: center;
    }

    /* 上传区域 */
    .upload-box {
        border: 2px dashed #d0d0d0;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        background-color: #fafafa;
        margin-bottom: 20px;
    }

    .upload-icon {
        font-size: 48px;
        color: #666;
    }

    .upload-text {
        color: #666;
        font-size: 14px;
        margin-top: 10px;
    }

    /* 文件信息 */
    .file-info {
        background-color: #e8f5e9;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .file-icon {
        font-size: 24px;
    }

    .file-name {
        font-weight: 500;
        color: #2e7d32;
    }

    /* 对话区域 */
    .chat-area {
        background-color: #f5f5f5;
        border-radius: 12px;
        padding: 20px;
        min-height: 300px;
    }

    /* 用户消息 */
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 15px;
        display: inline-block;
        max-width: 100%;
    }

    .user-label {
        font-size: 12px;
        color: #1976d2;
        font-weight: 600;
        margin-bottom: 5px;
    }

    /* AI消息 */
    .ai-message {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    .ai-label {
        font-size: 12px;
        color: #7b1fa2;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .ai-content {
        line-height: 1.8;
        color: #333;
    }

    /* 来源信息 */
    .source-box {
        background-color: #f0f0f0;
        padding: 10px 12px;
        border-radius: 8px;
        margin-top: 12px;
        font-size: 12px;
    }

    .source-title {
        color: #666;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .source-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
        padding: 6px 10px;
        background-color: white;
        border-radius: 6px;
    }

    .source-page {
        background-color: #1976d2;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }

    .source-chunk {
        background-color: #7b1fa2;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }

    .source-score {
        background-color: #4caf50;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
    }

    .source-file {
        color: #388e3c;
        font-size: 11px;
    }

    /* 输入框 */
    .stChatInput {
        margin-top: 15px;
    }

    /* 清除按钮 */
    .clear-btn {
        float: right;
    }
</style>
""", unsafe_allow_html=True)


def check_backend():
    """检查后端是否运行"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    # 标题
    st.markdown("<div class='main-title'>📄 ChatPDF</div>", unsafe_allow_html=True)

    # 后端状态
    if not check_backend():
        st.error("❌ 后端未连接，请启动后端服务")
        st.code("conda activate zhihu && python -m uvicorn backend_api:app --port 8000")
        st.stop()

    # 初始化 session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "filename" not in st.session_state:
        st.session_state.filename = None

    # 文件上传区域
    if not st.session_state.session_id:
        st.markdown("""
        <div class='upload-box'>
            <div class='upload-icon'>📁</div>
            <div class='upload-text'>上传 PDF 文档开始对话</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "选择 PDF 文件",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.button("上传并处理", type="primary", key="upload_btn"):
                with st.spinner("正在处理文档..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        response = requests.post(
                            f"{API_BASE_URL}/upload",
                            files=files,
                            timeout=180
                        )

                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.session_id = data["session_id"]
                            st.session_state.filename = data["filename"]
                            st.session_state.messages = []
                            st.success(f"✅ 文档已加载")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"处理失败: {response.text}")
                    except Exception as e:
                        st.error(f"出错: {str(e)}")
    else:
        # 显示已上传文件
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class='file-info'>
                <span class='file-icon'>📄</span>
                <span class='file-name'>{st.session_state.filename}</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("清除", key="clear_btn"):
                try:
                    requests.delete(f"{API_BASE_URL}/document/{st.session_state.session_id}")
                except:
                    pass
                st.session_state.session_id = None
                st.session_state.filename = None
                st.session_state.messages = []
                st.rerun()

        # 对话区域
        st.markdown("<div class='chat-area'>", unsafe_allow_html=True)

        # 显示历史消息
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='user-message'>
                    <div class='user-label'>👤 你</div>
                    <div>{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ai-message'>
                    <div class='ai-label'>🤖 AI</div>
                    <div class='ai-content'>{msg["content"]}</div>
                """, unsafe_allow_html=True)

                # 显示来源信息
                if msg.get("sources"):
                    st.markdown("""
                    <div class='source-box'>
                        <div class='source-title'>📚 参考来源</div>
                    """, unsafe_allow_html=True)

                    for source in msg["sources"]:
                        page = source.get("page", "?")
                        chunk = source.get("chunk_num", "?")
                        score = source.get("relevance_score", 0)
                        file = source.get("file", "document.pdf")
                        text_preview = source.get("text", "")

                        st.markdown(f"""
                        <div class='source-item'>
                            <span class='source-page'>P{page}</span>
                            <span class='source-chunk'>#{chunk}</span>
                            <span class='source-score'>距离:{score:.2f}</span>
                            <span class='source-file'>{file}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        # 显示 chunk 原文
                        st.markdown(f"<div style='font-size: 13px; color: #555; margin: 8px 0 15px 0; line-height: 1.6;'>{text_preview[:300]}{'...' if len(text_preview) > 300 else ''}</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 输入区域
        if prompt := st.chat_input("输入你的问题..."):
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 调用 API
            with st.spinner("思考中..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": prompt,
                            "top_k": 3
                        },
                        timeout=60
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # 添加助手消息（包含来源信息）
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["highlighted_chunks"]
                        })

                        st.rerun()
                    else:
                        st.error(f"查询失败: {response.text}")
                except Exception as e:
                    st.error(f"出错: {str(e)}")


if __name__ == "__main__":
    main()