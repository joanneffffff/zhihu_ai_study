# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - Streamlit Web界面

运行方式: streamlit run 8-disney_web_app.py
"""
import os
import sys
import json
import streamlit as st
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# 获取脚本所在目录，用于构建绝对路径
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果 __file__ 不可用，使用当前工作目录
    SCRIPT_DIR = os.getcwd()

# 页面配置
st.set_page_config(
    page_title="迪士尼RAG助手",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    st.error("请设置 DASHSCOPE_API_KEY 环境变量")
    st.stop()

dashscope.api_key = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-flash-2026-03-06"
INDEX_FILE = os.path.join(SCRIPT_DIR, "disney_full_index.faiss")
METADATA_FILE = os.path.join(SCRIPT_DIR, "disney_full_metadata.json")

# 问题类型
QUESTION_TYPES = {
    "票价": ["票价", "门票", "多少钱", "价格", "费用", "收费"],
    "优惠": ["优惠", "折扣", "特价", "便宜", "省钱", "活动"],
    "游玩": ["游玩", "项目", "设施", "排队", "体验", "攻略"],
    "餐饮": ["餐饮", "吃", "餐厅", "美食", "小吃", "用餐"],
    "住宿": ["住宿", "酒店", "入住", "房间", "度假"],
    "交通": ["交通", "怎么去", "地铁", "巴士", "停车"],
    "退改": ["退款", "退票", "改期", "取消", "变更"],
    "服务": ["服务", "VIP", "礼宾", "尊享", "快速通道"]


}


@st.cache_resource
def load_index():
    """加载索引（缓存）"""
    if not os.path.exists(INDEX_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata


def get_text_embedding(text):
    """获取文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def detect_question_type(query):
    """检测问题类型"""
    for qtype, keywords in QUESTION_TYPES.items():
        if any(kw in query for kw in keywords):
            return qtype
    return "通用"


def search(query, index, metadata, top_k=5):
    """检索相关内容"""
    query_vec = np.array([get_text_embedding(query)]).astype('float32')
    distances, indices = index.search(query_vec, min(top_k * 2, index.ntotal))

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        m = metadata[idx]
        results.append({
            "idx": int(idx),
            "distance": float(dist),
            "similarity": 1 / (1 + float(dist)),
            "metadata": m
        })
        if len(results) >= top_k:
            break

    return results


def find_related_images(text_results, query, metadata):
    """基于标签(tags)匹配相关图片

    通用方法：
    1. 从查询中提取主要主题标签
    2. 找有相同主题标签的图片
    """
    matched_images = []

    # 定义查询关键词到标签的映射
    query_to_tags = {
        "邮轮": "邮轮",
        "游轮": "邮轮",
        "门票": "门票",
        "票价": "门票",
        "价格": "价格",
        "酒店": "酒店",
        "攻略": "攻略",
    }

    # 从查询中提取主要主题标签
    query_tags = set()
    for keyword, tag in query_to_tags.items():
        if keyword in query:
            query_tags.add(tag)

    # 如果查询中没有明确的主题标签，从文本结果中提取
    if not query_tags:
        for r in text_results[:3]:
            tags = r["metadata"].get("tags", [])
            query_tags.update(tags[:3])

    # 通过tags过滤图片
    for m in metadata:
        if m["type"] == "image":
            img_tags = set(m.get("tags", []))
            if img_tags & query_tags:
                matched_images.append({
                    "distance": 0,
                    "similarity": 1.0,
                    "metadata": m
                })

    return matched_images[:3]


def generate_answer_stream(query, context, question_type):
    """生成答案（流式输出）"""
    system_prompt = f"""你是一个专业的迪士尼客服助手，正在回答游客关于{question_type}方面的问题。

请遵循以下原则：
1. 基于提供的背景知识准确回答，不要编造信息
2. 回答要简洁明了，重点突出
3. 如果涉及具体数字（价格、时间等），请准确引用
4. 如果背景知识中没有相关信息，请诚实告知
5. 保持友好专业的语气"""

    user_prompt = f"""背景知识：
{context}

用户问题：{query}

请根据背景知识回答用户问题："""

    stream = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
        stream=True
    )

    return stream


def main():
    # 标题
    st.title("🏰 迪士尼RAG助手")
    st.markdown("---")

    # 加载索引
    index, metadata = load_index()

    if index is None:
        st.warning("⚠️ 知识库索引未找到，请先运行 `6-disney_build_full_index.py` 构建索引")
        st.code("python 6-disney_build_full_index.py")
        return

    # 侧边栏
    with st.sidebar:
        st.header("📊 知识库统计")
        st.metric("总记录数", len(metadata))

        # 按类型统计
        type_counts = {}
        for m in metadata:
            t = m.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        col1, col2 = st.columns(2)
        with col1:
            st.metric("文本条目", type_counts.get("text", 0))
        with col2:
            st.metric("图片", type_counts.get("image", 0))

        st.markdown("---")

        # 问题类型选择
        st.header("🏷️ 问题类型")
        selected_type = st.selectbox(
            "选择问题类型",
            ["全部"] + list(QUESTION_TYPES.keys())
        )

        st.markdown("---")

        # 快捷问题
        st.header("💡 快捷问题")
        quick_questions = [
            "迪士尼门票多少钱？",
            "老人票有什么优惠？",
            "年卡有什么权益？",
            "入园有什么注意事项？",
            "如何购买VIP服务？"
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q}"):
                st.session_state["query_input"] = q

    # 主界面
    st.markdown("### 请输入您的问题")
    query = st.text_input(
        "查询",
        value=st.session_state.get("query_input", ""),
        placeholder="例如：迪士尼游轮价格",
        key="query_input",
        label_visibility="collapsed"
    )

    # 搜索按钮
    if st.button("🔍 搜索", type="primary") or query:
        if query:
            # 检测问题类型
            question_type = detect_question_type(query)

            # 检索
            with st.spinner("正在检索..."):
                results = search(query, index, metadata, top_k=5)

            # 构建上下文
            context_parts = []
            for r in results:
                if r["metadata"]["type"] == "text":
                    context_parts.append(r["metadata"]["content"])
            context = "\n---\n".join(context_parts[:3])

            # 匹配相关图片（使用tags匹配）
            matched_images = find_related_images(results, query, metadata)

            # 两列布局显示结果
            col_ans, col_img = st.columns([2, 1])

            with col_ans:
                # 显示答案（流式输出）
                st.markdown("### 📝 回答")
                answer_placeholder = st.empty()
                full_answer = ""

                stream = generate_answer_stream(query, context, question_type)
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            full_answer += delta.content
                            answer_placeholder.markdown(full_answer)

                # 显示来源
                st.markdown("### 📚 参考来源")
                for i, r in enumerate(results[:3]):
                    m = r["metadata"]
                    with st.expander(f"来源 {i + 1}: {m['source']} (相似度: {r['similarity']:.2%})"):
                        st.text(m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"])

            with col_img:
                st.markdown("### 🖼️ 相关图片")
                if matched_images:
                    for img in matched_images:
                        img_path = img["metadata"].get("path", "")
                        # 构建绝对路径
                        if img_path and not os.path.isabs(img_path):
                            img_path = os.path.join(SCRIPT_DIR, img_path)
                        if img_path and os.path.exists(img_path):
                            try:
                                image = Image.open(img_path)
                                st.image(image, caption=img["metadata"]["source"], use_container_width=True)
                            except Exception as e:
                                st.warning(f"无法加载图片: {img['metadata']['source']}")
                        else:
                            st.warning(f"图片文件不存在: {img['metadata']['source']}")
                else:
                    st.info("暂无相关图片")

    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "迪士尼RAG助手 | 基于多模态Embedding + FAISS + 大语言模型"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
