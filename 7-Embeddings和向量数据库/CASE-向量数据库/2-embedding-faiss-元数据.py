import os
import numpy as np
import faiss
import urllib3
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 禁用 SSL 警告和代理
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'

load_dotenv()

# 本地模型缓存路径
LOCAL_MODEL_PATH = "./models/Xorbits/bge-m3"

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

# Step1. 初始化 embedding 模型
try:
    embeddings = get_embeddings()
except Exception as e:
    print("初始化embedding模型失败。")
    print(f"错误信息: {e}")
    exit()

# Step2. 准备示例文本和元数据
documents = [
    {
        "id": "doc1",
        "text": "迪士尼乐园的门票一经售出，原则上不予退换。但在特殊情况下，如恶劣天气导致园区关闭，可在官方指引下进行改期或退款。",
        "metadata": {"source": "official_faq_v1.pdf", "category": "退票政策", "author": "Admin"}
    },
    {
        "id": "doc2",
        "text": "购买\"奇妙年卡\"的用户，可以享受一年内多次入园的特权，并且在餐饮和购物时有折扣。",
        "metadata": {"source": "annual_pass_rules.docx", "category": "会员权益", "author": "MarketingDept"}
    },
    {
        "id": "doc3",
        "text": "对于在线购买的迪士尼门票，如果需要退票，必须在票面日期前48小时通过原购买渠道提交申请，并可能收取手续费。",
        "metadata": {"source": "online_policy.html", "category": "退票政策", "author": "E-commerceTeam"}
    },
    {
        "id": "doc4",
        "text": "园区内的\"加勒比海盗\"项目因年度维护，将于下周暂停开放。",
        "metadata": {"source": "maintenance_notice.txt", "category": "园区公告", "author": "OpsDept"}
    }
]

# Step3. 创建元数据存储和向量列表
metadata_store = []
vectors_list = []
vector_ids = []

print("正在为文档生成向量...")
for i, doc in enumerate(documents):
    try:
        # 使用本地模型生成向量
        vector = embeddings.embed_query(doc["text"])
        vectors_list.append(vector)

        # 存储元数据
        metadata_store.append(doc)
        vector_ids.append(i)

        print(f"  - 已处理文档 {i+1}/{len(documents)}")

    except Exception as e:
        print(f"处理文档 '{doc['id']}' 时出错: {e}")
        continue

# 将向量列表转换为NumPy数组
vectors_np = np.array(vectors_list).astype('float32')
vector_ids_np = np.array(vector_ids)

# Step4. 构建并填充 FAISS 索引
dimension = len(vectors_list[0])  # bge-m3 向量维度
k = 2

# 创建 L2 距离索引
index_flat_l2 = faiss.IndexFlatL2(dimension)

# 使用 IndexIDMap 包装
index = faiss.IndexIDMap(index_flat_l2)

# 添加向量和 ID
index.add_with_ids(vectors_np, vector_ids_np)

print(f"\nFAISS 索引已成功创建，共包含 {index.ntotal} 个向量。")
print(f"向量维度: {dimension}")


# Step5. 执行搜索并检索元数据
query_text = "我想了解一下迪士尼门票的退款流程"
print(f"\n正在为查询文本生成向量: '{query_text}'")

try:
    # 为查询文本生成向量
    query_vector = np.array([embeddings.embed_query(query_text)]).astype('float32')

    # 在 FAISS 索引中执行搜索
    distances, retrieved_ids = index.search(query_vector, k)

    # Step6. 展示结果
    print("\n--- 搜索结果 ---")
    for i in range(k):
        doc_id = retrieved_ids[0][i]

        if doc_id == -1:
            print(f"\n排名 {i+1}: 未找到更多结果。")
            continue

        retrieved_doc = metadata_store[doc_id]

        print(f"\n--- 排名 {i+1} (相似度得分/距离: {distances[0][i]:.4f}) ---")
        print(f"ID: {doc_id}")
        print(f"原始文本: {retrieved_doc['text']}")
        print(f"元数据: {retrieved_doc['metadata']}")

except Exception as e:
    print(f"执行搜索时发生错误: {e}")