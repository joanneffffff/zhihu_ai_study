#!/usr/bin/env python
# coding: utf-8

# In[5]:


from google import genai

client = genai.Client()
# 文字输出
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="用中文解释AI大模型是如何工作的",
)

print(response.text)


# In[6]:


from PIL import Image
# 图像理解
image = Image.open("dog_and_girl.jpeg")

# 注意：contents 变成了一个列表，里面同时放了图片对象和文字
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[image, "帮我解释下这张照片"]
)

print(response.text)


# In[8]:


# 视频理解
import time

# 1. 上传视频文件
print("正在上传视频...")
video_file = client.files.upload(file="car.mp4") # 汽车剐蹭视频
print(f"上传成功: {video_file.name}")

# 2. 等待视频处理 (关键步骤！)
# 视频上传后，Google 需要几秒钟在云端进行转码。
while video_file.state.name == "PROCESSING":
    print("视频处理中，请稍候...")
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError("视频处理失败")

print("视频就绪，开始推理...")

# 3. 多模态推理
# 将上传好的 video_file 对象直接放入 contents 列表
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        video_file, 
        "详细描述视频里发生了什么？如果有对话，请把关键对话提取出来。"
    ]
)

print(response.text)

# 4. (可选) 删除云端文件以节省空间
# client.files.delete(name=video_file.name)

