from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from typing import List
import os
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

# 创建嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)

# 加载向量数据库
vectorstore = FAISS.load_local("./vector_db", embeddings, allow_dangerous_deserialization=True)

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

def multi_query_search(query: str, vectorstore, k: int = 4) -> List:
    """执行多查询检索，合并去重结果"""
    queries = generate_multi_queries(query)
    print(f"生成的查询变体: {queries}")

    seen_contents = set()
    unique_docs = []

    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

    return unique_docs

# 示例查询
query = "客户经理的考核标准是什么？"
# 执行查询
results = multi_query_search(query, vectorstore)

# 打印结果
print(f"\n查询: {query}")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)