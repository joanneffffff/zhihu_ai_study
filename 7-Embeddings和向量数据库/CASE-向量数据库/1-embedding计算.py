import os
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

# 获取 embedding 模型
embeddings = get_embeddings()

# 测试 embedding
text = '我想知道迪士尼的退票政策'
vector = embeddings.embed_query(text)

print(f"文本: {text}")
print(f"向量维度: {len(vector)}")
print(f"向量前10个值: {vector[:10]}")