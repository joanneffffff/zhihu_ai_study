"""
FastAPI 后端 - RAG 文档问答系统 API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle
import requests
import urllib3
import hashlib
import shutil
import tempfile
import uuid

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

app = FastAPI(title="RAG 文档问答系统 API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局存储
sessions = {}  # session_id -> {knowledge_base, page_info, original_text, chunks}


class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    highlighted_chunks: List[dict]


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


def get_embeddings():
    """获取嵌入模型"""
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


def extract_text_with_page_numbers(pdf) -> tuple:
    """从PDF中提取文本并记录页码"""
    text = ""
    page_numbers = []
    page_texts = []  # 保存每页的原始文本

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            page_texts.append({"page": page_number, "text": extracted_text})
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))

    return text, page_numbers, page_texts


def process_pdf(pdf_file) -> dict:
    """处理 PDF 文件，返回向量数据库和相关信息"""
    pdf_reader = PdfReader(pdf_file)
    text, page_numbers, page_texts = extract_text_with_page_numbers(pdf_reader)

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    embeddings = get_embeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # 记录页码信息
    lines = text.split("\n")
    page_info = {}
    chunk_page_map = {}  # chunk -> page number

    for i, chunk in enumerate(chunks):
        start_idx = text.find(chunk[:100])
        if start_idx == -1:
            for j, line in enumerate(lines):
                if chunk.startswith(line[:min(50, len(line))]):
                    start_idx = j
                    break
            if start_idx == -1:
                for j, line in enumerate(lines):
                    if line and line in chunk:
                        start_idx = text.find(line)
                        break

        if start_idx != -1:
            line_count = text[:start_idx].count("\n")
            if line_count < len(page_numbers):
                page_num = page_numbers[line_count]
            else:
                page_num = page_numbers[-1] if page_numbers else 1
        else:
            page_num = -1

        page_info[chunk] = page_num
        chunk_page_map[i] = page_num

    knowledge_base.page_info = page_info

    return {
        "knowledge_base": knowledge_base,
        "page_info": page_info,
        "original_text": text,
        "chunks": chunks,
        "chunk_page_map": chunk_page_map,
        "page_texts": page_texts
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并处理 PDF 文档"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 生成 session ID
    session_id = str(uuid.uuid4())

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 处理 PDF
        result = process_pdf(tmp_path)
        result["filename"] = file.filename  # 保存文件名
        sessions[session_id] = result

        return {
            "session_id": session_id,
            "filename": file.filename,
            "total_chunks": len(result["chunks"]),
            "total_pages": len(result["page_texts"])
        }
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """查询文档"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[request.session_id]
    kb = session["knowledge_base"]

    # 搜索相关文档，返回距离分数
    docs_with_scores = kb.similarity_search_with_score(request.question, k=request.top_k)

    # 构建上下文
    context = "\n\n".join([doc.page_content for doc, score in docs_with_scores])

    # 调用 LLM
    prompt = f"""根据以下上下文回答问题。如果上下文中没有相关信息，请说"根据文档内容，我无法回答这个问题"。

上下文:
{context}

问题: {request.question}

请用简洁、准确的语言回答:"""

    answer = call_llm(prompt)

    # 构建来源信息
    sources = []
    highlighted_chunks = []
    seen_pages = set()

    filename = session.get("filename", "document.pdf")

    for i, (doc, score) in enumerate(docs_with_scores):
        text_content = doc.page_content
        page_num = session["page_info"].get(text_content.strip(), -1)

        if page_num not in seen_pages:
            seen_pages.add(page_num)
            sources.append({
                "page": page_num,
                "text_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
            })

        # 高亮信息（包含真实的 FAISS 距离分数）
        # FAISS L2 距离：越小越相关
        highlighted_chunks.append({
            "text": text_content,
            "page": page_num,
            "chunk_num": i + 1,
            "file": filename,
            "relevance_score": round(float(score), 4)  # 真实的 FAISS 距离分数
        })

    return QueryResponse(
        answer=answer,
        sources=sources,
        highlighted_chunks=highlighted_chunks
    )


@app.get("/document/{session_id}")
async def get_document_info(session_id: str):
    """获取文档信息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "total_chunks": len(session["chunks"]),
        "total_pages": len(session["page_texts"]),
        "page_texts": session["page_texts"]
    }


@app.delete("/document/{session_id}")
async def delete_document(session_id: str):
    """删除文档会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"message": "Document deleted"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "active_sessions": len(sessions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
