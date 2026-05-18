#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 从环境变量获取配置
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')

# 模型下载
from modelscope import snapshot_download
model_name = 'BAAI/bge-reranker-large'
model_dir = snapshot_download(model_name, cache_dir=MODEL_CACHE_DIR)


# In[2]:


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

pairs = [['what is panda?', 'The giant panda is a bear species endemic to China.']]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数


# In[3]:


pairs = [
    ['what is panda?', 'The giant panda is a bear species endemic to China.'],  # 高相关
    ['what is panda?', 'Pandas are cute.'],                                     # 中等相关
    ['what is panda?', 'The Eiffel Tower is in Paris.']                        # 不相关
]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数

