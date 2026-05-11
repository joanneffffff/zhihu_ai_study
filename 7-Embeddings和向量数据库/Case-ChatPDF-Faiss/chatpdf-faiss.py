from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import List, Tuple, Optional, Any
import os
import pickle
import requests
import urllib3

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

API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')
if not API_KEY:
    raise ValueError("请设置环境变量 API_KEY")

# 本地模型缓存路径
LOCAL_MODEL_PATH = "./models/Xorbits/bge-m3"


class CustomLLM(LLM):
    """自定义 LLM 类，使用 requests 直接调用 API"""

    model: str = "glm-5.1"

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = BASE_URL.rstrip('/') + '/chat/completions'
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        if stop:
            data['stop'] = stop

        response = requests.post(url, headers=headers, json=data, timeout=60, verify=False)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']


def get_embeddings():
    """获取嵌入模型，优先使用本地缓存的模型"""
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        print(f"使用本地缓存的模型: {LOCAL_MODEL_PATH}")
        model_path = LOCAL_MODEL_PATH
    else:
        print("从 ModelScope 下载 bge-m3 模型...")
        from modelscope import snapshot_download
        model_path = snapshot_download(
            'Xorbits/bge-m3',
            cache_dir='./models'
        )
        print(f"模型已下载到: {model_path}")

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """从PDF中提取文本并记录每行文本对应的页码"""
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))

    return text, page_numbers


def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
    """处理文本并创建向量存储"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    embeddings = get_embeddings()
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


def load_knowledge_base(load_path: str, embeddings=None) -> FAISS:
    """从磁盘加载向量数据库和页码信息"""
    if embeddings is None:
        embeddings = get_embeddings()

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


# 主程序
if __name__ == "__main__":
    # 获取嵌入模型
    embeddings = get_embeddings()

    save_dir = "./vector_db"
    # 检查向量数据库是否已存在
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "index.faiss")):
        print(f"向量数据库已存在，从 {save_dir} 加载...")
        knowledgeBase = load_knowledge_base(save_dir, embeddings)
    else:
        pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
        text, page_numbers = extract_text_with_page_numbers(pdf_reader)
        print(f"提取的文本长度: {len(text)} 个字符。")
        knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=save_dir)

    # 创建 LLM
    llm = CustomLLM(model="glm-5.1")

    # 设置查询问题
    query = "客户经理被投诉了，投诉一次扣多少分"
    if query:
        docs = knowledgeBase.similarity_search(query, k=2)

        # 构建 prompt
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""根据以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文:
{context}

问题: {query}

答案:"""

        print("正在查询 LLM...")
        response = llm.invoke(prompt)
        print(f"\n回答:\n{response}")
        print("\n来源:")

        unique_pages = set()
        for doc in docs:
            text_content = getattr(doc, "page_content", "")
            source_page = knowledgeBase.page_info.get(text_content.strip(), "未知")
            if source_page not in unique_pages:
                unique_pages.add(source_page)
                print(f"文本块页码: {source_page}")
