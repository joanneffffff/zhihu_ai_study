# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 完整版查询系统

功能：
1. 加载完整知识库索引
2. 多模态检索（文本、图片）
3. 智能意图识别（问题分类）
4. RAG问答生成
5. 支持对话历史
"""
import os
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-flash-2026-03-06"
INDEX_FILE = "disney_full_index.faiss"
METADATA_FILE = "disney_full_metadata.json"

# 意图关键词配置
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图", "样子", "外观"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]

# 媒体匹配阈值（图片距离通常比文本大）
MEDIA_DISTANCE_THRESHOLD = 2.0

# 图片关键词（用于检测是否需要图片）
IMAGE_INTENT_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图", "样子", "外观", "价格表", "价目表"]

# 问题类型关键词
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

class DisneyRAGAssistant:
    """迪士尼RAG助手类"""

    def __init__(self, index_file=INDEX_FILE, metadata_file=METADATA_FILE):
        self.index = None
        self.metadata = None
        self.conversation_history = []
        self.load_index(index_file, metadata_file)

    def load_index(self, index_file, metadata_file):
        """加载索引和元数据"""
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"索引文件不存在: {index_file}，请先运行构建脚本")

        self.index = faiss.read_index(index_file)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"已加载索引: {self.index.ntotal} 条记录")

    def get_text_embedding(self, text):
        """获取文本embedding"""
        resp = dashscope.MultiModalEmbedding.call(
            model=MULTIMODAL_EMBEDDING_MODEL,
            input=[{'text': text}]
        )
        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"Embedding失败: {resp.message}")
        return resp.output['embeddings'][0]['embedding']

    def distance_to_similarity(self, distance):
        """L2距离转相似度"""
        return 1 / (1 + distance)

    def detect_question_type(self, query):
        """检测问题类型"""
        query_lower = query.lower()
        detected_types = []

        for qtype, keywords in QUESTION_TYPES.items():
            if any(kw in query_lower for kw in keywords):
                detected_types.append(qtype)

        return detected_types if detected_types else ["通用"]

    def detect_media_intent(self, query):
        """检测媒体意图"""
        query_lower = query.lower()
        want_image = any(kw in query_lower for kw in IMAGE_INTENT_KEYWORDS)
        want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
        return want_image, want_video

    def find_related_images(self, text_results, query):
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
                # 只取前3个标签，避免噪声
                query_tags.update(tags[:3])

        # 通过tags过滤图片
        for m in self.metadata:
            if m["type"] == "image":
                img_tags = set(m.get("tags", []))
                # 检查图片tags是否与查询tags有交集
                if img_tags & query_tags:
                    matched_images.append({
                        "distance": 0,
                        "similarity": 1.0,
                        "metadata": m
                    })

        return matched_images[:3]  # 最多返回3张图片

    def search(self, query, top_k=5, filters=None):
        """检索相关内容"""
        query_vec = np.array([self.get_text_embedding(query)]).astype('float32')

        # 检索更多结果用于过滤
        distances, indices = self.index.search(query_vec, min(top_k * 3, self.index.ntotal))

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            m = self.metadata[idx]

            # 应用过滤器
            if filters:
                if filters.get("type") and m["type"] != filters["type"]:
                    continue
                if filters.get("category") and m.get("category") != filters["category"]:
                    continue

            results.append({
                "idx": idx,
                "distance": float(dist),
                "similarity": self.distance_to_similarity(float(dist)),
                "metadata": m
            })

            if len(results) >= top_k:
                break

        return results

    def format_context(self, results, max_length=3000):
        """格式化上下文"""
        context_parts = []
        total_length = 0

        for i, r in enumerate(results):
            m = r["metadata"]
            if m["type"] != "text":
                continue

            part = f"【来源: {m['source']}】\n{m['content']}\n"
            if total_length + len(part) > max_length:
                break

            context_parts.append(part)
            total_length += len(part)

        return "\n---\n".join(context_parts)

    def generate_answer(self, query, context, question_types):
        """生成答案"""
        type_hint = "、".join(question_types) if question_types else "通用"

        system_prompt = f"""你是一个专业的迪士尼客服助手，正在回答游客关于{type_hint}方面的问题。

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

        response = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def ask(self, query, top_k=5, show_details=True):
        """RAG问答主函数"""
        # 检测意图
        question_types = self.detect_question_type(query)
        want_image, want_video = self.detect_media_intent(query)

        if show_details:
            print(f"\n{'=' * 60}")
            print(f"问题: {query}")
            print(f"问题类型: {', '.join(question_types)}")
            print(f"媒体意图: 图片={want_image}, 视频={want_video}")
            print("=" * 60)

        # 检索文本结果
        text_results = self.search(query, top_k=top_k, filters={"type": "text"})

        if show_details:
            print(f"\n检索结果 (Top-{len(text_results)}):")
            print("-" * 60)
            for i, r in enumerate(text_results[:5]):
                m = r["metadata"]
                content_preview = m["content"][:60].replace("\n", " ")
                print(f"  [{i + 1}] {m['source']} (相似度: {r['similarity']:.4f})")
                print(f"      {content_preview}...")

        # 构建上下文
        context = self.format_context(text_results)

        # 生成答案
        answer = self.generate_answer(query, context, question_types)

        # 智能匹配相关图片（基于文本结果和查询关键词）
        matched_images = self.find_related_images(text_results, query)

        # 组装最终答案
        final_answer = answer

        if matched_images:
            final_answer += "\n\n📷 相关图片："
            for img in matched_images:
                final_answer += f"\n  - {img['metadata']['path']}"

        if show_details:
            print(f"\n回答:")
            print("-" * 60)
            print(final_answer)

        # 记录对话历史
        self.conversation_history.append({
            "query": query,
            "answer": final_answer,
            "sources": [r["metadata"]["source"] for r in text_results[:3]]
        })

        return {
            "query": query,
            "answer": final_answer,
            "sources": text_results[:3],
            "images": matched_images,
            "question_types": question_types
        }

    def interactive_mode(self):
        """交互式问答模式"""
        print("\n" + "=" * 60)
        print("迪士尼RAG助手 - 交互式问答")
        print("输入问题进行咨询，输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        while True:
            try:
                query = input("\n请输入问题: ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break

                self.ask(query)

            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    """主函数"""
    # 初始化助手
    assistant = DisneyRAGAssistant()

    # 测试查询
    print("\n" + "=" * 60)
    print("测试查询")
    print("=" * 60)

    test_queries = [
        "迪士尼门票多少钱？",
        "有什么优惠活动吗？",
        "老人票有什么规定？",
        "年卡有什么权益？",
        "入园有什么注意事项？",
    ]

    for query in test_queries:
        assistant.ask(query)
        print()

    # 进入交互模式
    print("\n进入交互模式...")
    assistant.interactive_mode()


if __name__ == "__main__":
    main()
