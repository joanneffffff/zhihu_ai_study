from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from typing import List, Tuple, Set
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
    """
    从PDF中提取文本并记录每行文本对应的页码
    """
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

def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    """
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
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
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
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    """
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
    else:
        print("警告: 未找到页码信息文件。")

    return knowledgeBase

def generate_multi_queries(query: str, num_queries: int = 3) -> List[str]:
    """
    使用LLM生成多个查询变体
    """
    prompt = f"""你是一个AI助手，负责生成多个不同视角的搜索查询。
给定一个用户问题，生成{num_queries}个不同但相关的查询，以帮助检索更全面的信息。
每个查询应该从不同角度表达相同的信息需求。

原始问题: {query}

请直接输出{num_queries}个查询，每行一个，不要编号和其他内容:"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    queries = [q.strip() for q in content.strip().split('\n') if q.strip()]
    queries = [query] + queries[:num_queries]
    return queries

def multi_query_search(query: str, vectorstore, k: int = 4) -> List:
    """
    执行多查询检索，合并去重结果
    """
    queries = generate_multi_queries(query)
    print(f"生成的查询变体: {queries}")

    seen_contents = set()
    unique_docs = []

    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        for doc in docs:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)

    return unique_docs

def process_query_with_multi_retriever(query: str, vectorstore) -> Tuple[str, Set]:
    """
    使用多查询检索处理查询
    """
    docs = multi_query_search(query, vectorstore)
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
        text_content = doc.page_content
        source_page = vectorstore.page_info.get(text_content.strip(), "未知")
        unique_pages.add(source_page)

    return response.choices[0].message.content, unique_pages

def main():
    pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
    vector_db_path = './vector_db'

    if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
        print(f"发现现有向量数据库: {vector_db_path}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True}
        )
        knowledgeBase = load_knowledge_base(vector_db_path, embeddings)
    else:
        print(f"未找到向量数据库，将从PDF创建新的向量数据库")
        pdf_reader = PdfReader(pdf_path)
        text, page_numbers = extract_text_with_page_numbers(pdf_reader)
        print(f"提取的文本长度: {len(text)} 个字符。")
        knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=vector_db_path)

    queries = [
        "客户经理被投诉了，投诉一次扣多少分",
        "客户经理每年评聘申报时间是怎样的？",
        "客户经理的考核标准是什么？"
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"查询: {query}")

        response, unique_pages = process_query_with_multi_retriever(
            query,
            knowledgeBase
        )

        print("\n回答:")
        print(response)

        print("\n来源页码:")
        for page in sorted(unique_pages):
            print(f"- 第 {page} 页")
        print("="*50)

if __name__ == "__main__":
    main()