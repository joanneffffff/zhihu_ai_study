# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 完整知识库构建

功能：
1. 递归扫描知识库目录，支持多级子目录
2. 支持 DOCX、PDF、TXT 文件解析
3. 支持图片、视频多模态处理
4. 智能切片策略（按段落/语义切分）
5. 构建FAISS向量索引
"""
import os
import base64
import json
import re
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from docx import Document as DocxDocument
from dotenv import load_dotenv

load_dotenv()

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

# 知识库根目录
KNOWLEDGE_BASE_DIR = "迪士尼RAG知识库（完整）"
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-flash-2026-03-06"

# 输出文件
INDEX_FILE = "disney_full_index.faiss"
METADATA_FILE = "disney_full_metadata.json"

# 切分参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100

# 支持的文件类型
SUPPORTED_DOC_EXTENSIONS = {'.docx', '.pdf', '.txt'}
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


def parse_docx(file_path):
    """解析 DOCX 文件，提取文本和表格"""
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif element.tag.endswith('tbl'):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n\n".join(all_text)


def parse_pdf(file_path):
    """解析 PDF 文件"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text.strip())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        print(f"警告: 未安装 PyMuPDF，跳过 PDF 文件: {file_path}")
        return ""
    except Exception as e:
        print(f"PDF解析失败 {file_path}: {e}")
        return ""


def parse_txt(file_path):
    """解析 TXT 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read().strip()


def smart_chunk_text(text, source_file, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """智能文本切分 - 按段落优先，保持语义完整性"""
    if not text or len(text) < MIN_CHUNK_SIZE:
        return []

    chunks = []

    # 按段落分割
    paragraphs = re.split(r'\n\s*\n', text)

    current_chunk = ""
    chunk_id = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果单个段落就超过chunk_size，需要进一步切分
        if len(para) > chunk_size:
            # 先保存当前累积的内容
            if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "content": current_chunk.strip(),
                    "source": source_file,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
                current_chunk = ""

            # 按句子切分大段落
            sentences = re.split(r'([。！？\n])', para)
            sentence_buffer = ""

            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                if len(sentence_buffer) + len(sentence) > chunk_size:
                    if sentence_buffer and len(sentence_buffer) >= MIN_CHUNK_SIZE:
                        chunks.append({
                            "content": sentence_buffer.strip(),
                            "source": source_file,
                            "chunk_id": chunk_id
                        })
                        chunk_id += 1
                    # 保留overlap部分
                    sentence_buffer = sentence_buffer[-overlap:] if overlap > 0 else ""
                sentence_buffer += sentence

            if sentence_buffer and len(sentence_buffer) >= MIN_CHUNK_SIZE:
                current_chunk = sentence_buffer
        else:
            # 正常段落处理
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source": source_file,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

    # 处理最后一个chunk
    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append({
            "content": current_chunk.strip(),
            "source": source_file,
            "chunk_id": chunk_id
        })

    return chunks


def get_text_embedding(text):
    """获取文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_image_embedding(image_path):
    """获取图片embedding"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    image_data = f"data:image/{ext};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'image': image_data}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_video_embedding(video_url):
    """获取视频embedding（多帧取平均）"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'video': video_url}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.message}")

    embeddings = resp.output['embeddings']
    if len(embeddings) > 1:
        vectors = [np.array(e['embedding']) for e in embeddings]
        return np.mean(vectors, axis=0).tolist()
    return embeddings[0]['embedding']


def extract_tags_from_filename(filename):
    """从文件名自动提取标签

    将文件名中的关键词转换为标准化的标签
    例如: "迪士尼邮轮价格1.jpeg" -> ["邮轮", "价格"]
    """
    # 定义关键词到标签的映射
    tag_mapping = {
        "邮轮": "邮轮",
        "游轮": "邮轮",
        "门票": "门票",
        "票价": "门票",
        "价格": "价格",
        "价目表": "价格",
        "酒店": "酒店",
        "住宿": "酒店",
        "攻略": "攻略",
        "地图": "地图",
        "海报": "海报",
        "活动": "活动",
        "万圣节": "节日活动",
        "圣诞节": "节日活动",
        "春节": "节日活动",
    }

    # 提取标签
    tags = []
    filename_lower = filename.lower()
    for keyword, tag in tag_mapping.items():
        if keyword in filename_lower:
            tags.append(tag)

    return tags


def extract_tags_from_text(text):
    """从文本内容自动提取标签

    分析文本内容，提取主题标签
    """
    # 定义关键词到标签的映射
    tag_mapping = {
        "邮轮": "邮轮",
        "游轮": "邮轮",
        "门票": "门票",
        "票价": "门票",
        "价格": "价格",
        "酒店": "酒店",
        "住宿": "酒店",
        "攻略": "攻略",
        "退款": "退款",
        "退票": "退款",
        "投诉": "投诉",
        "客服": "客服",
        "VIP": "VIP服务",
        "尊享": "VIP服务",
        "礼宾": "VIP服务",
        "年卡": "年卡",
        "老人": "老人票",
        "儿童": "儿童票",
        "餐饮": "餐饮",
        "餐厅": "餐饮",
    }

    # 提取标签（去重）
    tags = set()
    for keyword, tag in tag_mapping.items():
        if keyword in text:
            tags.add(tag)

    return list(tags)


def scan_knowledge_base(base_dir):
    """递归扫描知识库目录"""
    documents = []
    images = []
    videos = []

    for root, dirs, files in os.walk(base_dir):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            if filename.startswith('.'):
                continue

            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            # 获取相对路径作为分类
            rel_path = os.path.relpath(root, base_dir)
            category = rel_path if rel_path != '.' else "根目录"

            if ext in SUPPORTED_DOC_EXTENSIONS:
                documents.append({
                    "path": file_path,
                    "filename": filename,
                    "category": category,
                    "type": ext[1:]
                })
            elif ext in SUPPORTED_IMAGE_EXTENSIONS:
                images.append({
                    "path": file_path,
                    "filename": filename,
                    "category": category
                })
            elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                videos.append({
                    "path": file_path,
                    "filename": filename,
                    "category": category
                })

    return documents, images, videos


def build_and_save():
    """构建完整知识库索引"""
    print("\n" + "=" * 60)
    print("迪士尼RAG知识库 - 完整版构建")
    print("=" * 60)

    # 扫描知识库
    print("\n[1/5] 扫描知识库目录...")
    documents, images, videos = scan_knowledge_base(KNOWLEDGE_BASE_DIR)
    print(f"  发现文档: {len(documents)} 个")
    print(f"  发现图片: {len(images)} 个")
    print(f"  发现视频: {len(videos)} 个")

    metadata_store = []
    all_vectors = []
    doc_id = 0
    error_count = 0

    # 处理文档
    print("\n[2/5] 处理文档文件...")
    for i, doc_info in enumerate(documents):
        file_path = doc_info["path"]
        filename = doc_info["filename"]
        category = doc_info["category"]
        file_type = doc_info["type"]

        print(f"  [{i + 1}/{len(documents)}] {filename} ({category})")

        try:
            # 解析文档
            if file_type == 'docx':
                text = parse_docx(file_path)
            elif file_type == 'pdf':
                text = parse_pdf(file_path)
            elif file_type == 'txt':
                text = parse_txt(file_path)
            else:
                continue

            if not text:
                print(f"    警告: 文档内容为空，跳过")
                continue

            # 智能切分
            chunks = smart_chunk_text(text, filename)
            print(f"    文档长度: {len(text)} 字符, 切分为 {len(chunks)} 个chunk")

            # 获取embedding
            for chunk_info in chunks:
                # 从文件名和内容提取标签
                filename_tags = extract_tags_from_filename(filename)
                content_tags = extract_tags_from_text(chunk_info["content"])
                # 合并标签（去重）
                all_tags = list(set(filename_tags + content_tags))

                metadata = {
                    "id": doc_id,
                    "source": filename,
                    "category": category,
                    "type": "text",
                    "content": chunk_info["content"],
                    "chunk_id": chunk_info["chunk_id"],
                    "tags": all_tags  # 添加标签
                }

                vector = get_text_embedding(chunk_info["content"])
                all_vectors.append(vector)
                metadata_store.append(metadata)
                doc_id += 1

        except Exception as e:
            print(f"    错误: {e}")
            error_count += 1
            continue

    # 处理图片
    print("\n[3/5] 处理图片文件...")
    for i, img_info in enumerate(images):
        file_path = img_info["path"]
        filename = img_info["filename"]
        category = img_info["category"]

        print(f"  [{i + 1}/{len(images)}] {filename}")

        try:
            # 从文件名提取标签
            tags = extract_tags_from_filename(filename)

            metadata = {
                "id": doc_id,
                "source": filename,
                "category": category,
                "type": "image",
                "path": file_path,
                "content": f"[图片] {filename}",
                "tags": tags  # 添加标签
            }

            vector = get_image_embedding(file_path)
            all_vectors.append(vector)
            metadata_store.append(metadata)
            doc_id += 1

        except Exception as e:
            print(f"    错误: {e}")
            error_count += 1
            continue

    # 处理视频（本地视频暂不支持，需要URL）
    if videos:
        print("\n[4/5] 处理视频文件...")
        print(f"  注意: 多模态embedding暂不支持本地视频，跳过 {len(videos)} 个视频文件")

    # 创建FAISS索引
    print("\n[5/5] 构建向量索引...")
    if all_vectors:
        dim = len(all_vectors[0])
        print(f"  向量维度: {dim}")
        print(f"  总向量数: {len(all_vectors)}")

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors).astype('float32'))

        # 保存索引
        faiss.write_index(index, INDEX_FILE)
        print(f"  索引已保存: {INDEX_FILE}")

        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=2)
        print(f"  元数据已保存: {METADATA_FILE}")

    # 统计信息
    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)

    # 按类别统计
    category_stats = {}
    type_stats = {"text": 0, "image": 0, "video": 0}

    for m in metadata_store:
        type_stats[m["type"]] = type_stats.get(m["type"], 0) + 1
        cat = m.get("category", "未知")
        category_stats[cat] = category_stats.get(cat, 0) + 1

    print(f"\n按类型统计:")
    print(f"  文本chunk: {type_stats['text']}")
    print(f"  图片: {type_stats['image']}")
    print(f"  视频: {type_stats['video']}")

    print(f"\n按目录统计:")
    for cat, count in sorted(category_stats.items()):
        print(f"  {cat}: {count} 条")

    print(f"\n处理错误: {error_count} 个文件")

    return metadata_store


if __name__ == "__main__":
    build_and_save()
