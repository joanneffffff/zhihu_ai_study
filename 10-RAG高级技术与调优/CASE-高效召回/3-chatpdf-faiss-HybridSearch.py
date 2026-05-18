from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Set
import jieba
import os
import pickle
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 从环境变量中获取配置
API_KEY = os.getenv('ANTHROPIC_API_KEY')
BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.sfkey.cn/')
DEFAULT_MODEL = os.getenv('ANTHROPIC_MODEL', 'glm-5.1')
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', '../CASE-rerank/models/BAAI/bge-m3')

if not API_KEY:
    raise ValueError("请设置环境变量 ANTHROPIC_API_KEY")

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """从PDF中提取文本并记录每行文本对应的页码"""
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            print(f"No text found on page {page_number}.")

    return text, page_numbers

def tokenize_chinese(text: str) -> List[str]:
    """中文分词"""
    return list(jieba.cut(text))

class HybridRetriever:
    """混合检索器: BM25 + Vector"""

    def __init__(self, chunks: List[str], vectorstore: FAISS, alpha: float = 0.5):
        """
        初始化混合检索器

        参数:
            chunks: 文本块列表
            vectorstore: FAISS向量存储
            alpha: 向量检索权重 (0-1), BM25权重为 1-alpha
        """
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.alpha = alpha

        # 构建BM25索引
        tokenized_chunks = [tokenize_chinese(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # 构建chunk到索引的映射
        self.chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        执行混合检索

        参数:
            query: 查询文本
            k: 返回结果数量
        """
        # BM25检索
        tokenized_query = tokenize_chinese(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 归一化BM25分数
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

        # 向量检索 (获取更多结果用于融合)
        vector_results = self.vectorstore.similarity_search_with_score(query, k=len(self.chunks))

        # 构建向量分数字典 (距离越小越好，转换为分数)
        vector_scores = {}
        max_distance = max(score for _, score in vector_results) if vector_results else 1
        for doc, distance in vector_results:
            idx = self.chunk_to_idx.get(doc.page_content)
            if idx is not None:
                # 距离转分数: 1 - (distance / max_distance)
                vector_scores[idx] = 1 - (distance / max_distance) if max_distance > 0 else 0

        # 融合分数
        hybrid_scores = []
        for idx in range(len(self.chunks)):
            bm25_score = bm25_scores_normalized[idx]
            vector_score = vector_scores.get(idx, 0)
            combined = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            hybrid_scores.append((idx, combined))

        # 排序并返回top-k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = hybrid_scores[:k]

        results = []
        for idx, score in top_k:
            doc = Document(page_content=self.chunks[idx], metadata={"hybrid_score": score})
            results.append(doc)

        return results

def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> Tuple[FAISS, List[str]]:
    """处理文本并创建向量存储，同时返回chunks用于BM25"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True}
    )

    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")

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

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")

        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)

        with open(os.path.join(save_path, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        print(f"chunks已保存到: {os.path.join(save_path, 'chunks.pkl')}")

    return knowledgeBase, chunks

def load_knowledge_base(load_path: str, embeddings=None) -> Tuple[FAISS, List[str]]:
    """从磁盘加载向量数据库、页码信息和chunks"""
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True}
        )

    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")

    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")

    chunks_path = os.path.join(load_path, "chunks.pkl")
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        print("chunks已加载。")

    return knowledgeBase, chunks

def generate_multi_queries(query: str, num_queries: int = 3) -> List[str]:
    """使用LLM生成多个查询变体"""
    prompt = f"""你是一个AI助手，负责生成多个不同视角的搜索查询。
给定一个用户问题，生成{num_queries}个不同但相关的查询，以帮助检索更全面的信息。

原始问题: {query}

请直接输出{num_queries}个查询，每行一个，不要编号和其他内容:"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    queries = [q.strip() for q in content.strip().split('\n') if q.strip()]
    return [query] + queries[:num_queries]

def hybrid_multi_query_search(query: str, hybrid_retriever: HybridRetriever, k: int = 4) -> List[Document]:
    """混合检索 + 多查询"""
    queries = generate_multi_queries(query)
    print(f"生成的查询变体: {queries}")

    seen_contents = set()
    unique_docs = []

    for q in queries:
        docs = hybrid_retriever.search(q, k=k)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

    return unique_docs

def process_query(query: str, hybrid_retriever: HybridRetriever, vectorstore: FAISS) -> Tuple[str, Set]:
    """处理查询并返回回答"""
    docs = hybrid_multi_query_search(query, hybrid_retriever)
    print(f"找到 {len(docs)} 个相关文档")

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""根据以下上下文回答问题:

{context}

问题: {query}"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    unique_pages = set()
    for doc in docs:
        source_page = vectorstore.page_info.get(doc.page_content.strip(), "未知")
        unique_pages.add(source_page)

    return response.choices[0].message.content, unique_pages

def main():
    pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
    vector_db_path = './vector_db_hybrid'

    if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
        print(f"发现现有向量数据库: {vector_db_path}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True}
        )
        knowledgeBase, chunks = load_knowledge_base(vector_db_path, embeddings)
    else:
        print(f"未找到向量数据库，将从PDF创建新的向量数据库")
        pdf_reader = PdfReader(pdf_path)
        text, page_numbers = extract_text_with_page_numbers(pdf_reader)
        print(f"提取的文本长度: {len(text)} 个字符。")
        knowledgeBase, chunks = process_text_with_splitter(text, page_numbers, save_path=vector_db_path)

    # 创建混合检索器 (alpha=0.5 表示BM25和向量各占50%权重)
    hybrid_retriever = HybridRetriever(chunks, knowledgeBase, alpha=0.5)
    print("混合检索器已创建 (BM25 + Vector, alpha=0.5)")

    queries = [
        "客户经理被投诉了，投诉一次扣多少分",
        "客户经理每年评聘申报时间是怎样的？",
        "客户经理的考核标准是什么？"
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"查询: {query}")

        response, unique_pages = process_query(query, hybrid_retriever, knowledgeBase)

        print("\n回答:")
        print(response)

        print("\n来源页码:")
        for page in sorted(unique_pages):
            print(f"- 第 {page} 页")
        print("="*50)

if __name__ == "__main__":
    main()