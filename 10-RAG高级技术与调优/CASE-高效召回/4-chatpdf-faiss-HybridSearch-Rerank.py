from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from modelscope import AutoModelForSequenceClassification, AutoTokenizer, snapshot_download
from typing import List, Tuple, Set
import jieba
import torch
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

class Reranker:
    """基于ModelScope的Rerank模型"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", cache_dir: str = "./models"):
        """
        初始化Reranker

        参数:
            model_name: ModelScope上的模型名称
                - BAAI/bge-reranker-base (轻量级，推荐)
                - BAAI/bge-reranker-large (效果更好，但更慢)
            cache_dir: 模型缓存目录，默认为当前目录下的 ./models
        """
        print(f"正在加载Rerank模型: {model_name}")
        print(f"模型缓存目录: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

        # 先下载模型到指定目录
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        print(f"模型已下载到: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # 使用GPU如果可用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Rerank模型已加载，使用设备: {self.device}")

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        对文档进行重排序

        参数:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果，None表示返回全部
        """
        if not documents:
            return []

        # 构建query-document对
        pairs = [[query, doc.page_content] for doc in documents]

        # 批量计算分数
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

        # 如果只有一个文档，scores可能是float而不是list
        if isinstance(scores, float):
            scores = [scores]

        # 为每个文档添加rerank分数
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 更新metadata中的分数
        results = []
        for doc, score in scored_docs:
            doc.metadata["rerank_score"] = score
            results.append(doc)

        if top_k:
            results = results[:top_k]

        return results

class HybridRetriever:
    """混合检索器: BM25 + Vector"""

    def __init__(self, chunks: List[str], vectorstore: FAISS, alpha: float = 0.5):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.alpha = alpha

        tokenized_chunks = [tokenize_chinese(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.chunk_to_idx = {chunk: idx for idx, chunk in enumerate(chunks)}

    def search(self, query: str, k: int = 4) -> List[Document]:
        """执行混合检索"""
        tokenized_query = tokenize_chinese(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

        vector_results = self.vectorstore.similarity_search_with_score(query, k=len(self.chunks))

        vector_scores = {}
        max_distance = max(score for _, score in vector_results) if vector_results else 1
        for doc, distance in vector_results:
            idx = self.chunk_to_idx.get(doc.page_content)
            if idx is not None:
                vector_scores[idx] = 1 - (distance / max_distance) if max_distance > 0 else 0

        hybrid_scores = []
        for idx in range(len(self.chunks)):
            bm25_score = bm25_scores_normalized[idx]
            vector_score = vector_scores.get(idx, 0)
            combined = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            hybrid_scores.append((idx, combined))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = hybrid_scores[:k]

        results = []
        for idx, score in top_k:
            doc = Document(page_content=self.chunks[idx], metadata={"hybrid_score": score})
            results.append(doc)

        return results

def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> Tuple[FAISS, List[str]]:
    """处理文本并创建向量存储"""
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

    return knowledgeBase, chunks

def load_knowledge_base(load_path: str, embeddings=None) -> Tuple[FAISS, List[str]]:
    """从磁盘加载向量数据库"""
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

    chunks_path = os.path.join(load_path, "chunks.pkl")
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

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

def hybrid_multi_query_search_with_rerank(
    query: str,
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    initial_k: int = 10,
    final_k: int = 4
) -> List[Document]:
    """混合检索 + 多查询 + Rerank"""
    queries = generate_multi_queries(query)
    print(f"生成的查询变体: {queries}")

    # 第一阶段: 多查询混合检索，获取更多候选
    seen_contents = set()
    candidate_docs = []

    for q in queries:
        docs = hybrid_retriever.search(q, k=initial_k)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                candidate_docs.append(doc)

    print(f"初步召回 {len(candidate_docs)} 个候选文档")

    # 第二阶段: Rerank精排
    reranked_docs = reranker.rerank(query, candidate_docs, top_k=final_k)
    print(f"Rerank后保留 {len(reranked_docs)} 个文档")

    return reranked_docs

def process_query(
    query: str,
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    vectorstore: FAISS
) -> Tuple[str, Set]:
    """处理查询并返回回答"""
    docs = hybrid_multi_query_search_with_rerank(query, hybrid_retriever, reranker)

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

    # 创建混合检索器
    hybrid_retriever = HybridRetriever(chunks, knowledgeBase, alpha=0.5)
    print("混合检索器已创建 (BM25 + Vector)")

    # 创建Reranker (使用ModelScope上的轻量级模型)
    reranker = Reranker(model_name="BAAI/bge-reranker-base")

    queries = [
        "客户经理被投诉了，投诉一次扣多少分",
        # "客户经理每年评聘申报时间是怎样的？",
        # "客户经理的考核标准是什么？"
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"查询: {query}")

        response, unique_pages = process_query(query, hybrid_retriever, reranker, knowledgeBase)

        print("\n回答:")
        print(response)

        print("\n来源页码:")
        for page in sorted(unique_pages):
            print(f"- 第 {page} 页")
        print("="*50)

if __name__ == "__main__":
    main()