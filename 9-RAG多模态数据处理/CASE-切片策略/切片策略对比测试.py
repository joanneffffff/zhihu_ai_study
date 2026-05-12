#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略对比测试脚本
展示6种不同切片策略的效果对比
"""

import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. 固定长度切片
def improved_fixed_length_chunking(text, chunk_size=512, overlap=50):
    """固定长度切片 - 在句子边界切分"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界切分
        if end < len(text):
            # 寻找最近的句子结束符
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?。！？':
                    end = i + 1
                    break
        
        chunk = text[start:end]
        
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks

# 2. 句子边界切片
def semantic_chunking(text, max_chunk_size=512):
    """基于句子边界的切片 - 按句子分割"""
    # 使用正则表达式分割句子
    sentences = re.split(r'[.!?。！？\n]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果当前句子加入后超过最大长度，保存当前块
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # 添加最后一个块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# 3. LLM语义切片（LLM）
def advanced_semantic_chunking_with_llm(text, max_chunk_size=512):
    """使用LLM进行语义切片"""
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量，将使用基础语义切片")
        return semantic_chunking(text, max_chunk_size)
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    prompt = f"""
请将以下文本按照语义完整性进行切片，每个切片不超过{max_chunk_size}字符。
要求：
1. 保持语义完整性
2. 在自然的分割点切分
3. 返回JSON格式的切片列表，格式如下：
{{
  "chunks": [
    "第一个切片内容",
    "第二个切片内容",
    ...
  ]
}}

文本内容：
{text}

请返回JSON格式的切片列表：
"""
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个专业的文本切片助手。请严格按照JSON格式返回结果，不要添加任何额外的标记。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        # 清理结果，移除可能的Markdown代码块标记
        cleaned_result = result.strip()
        if cleaned_result.startswith('```'):
            cleaned_result = re.sub(r'^```(?:json)?\s*', '', cleaned_result)
        if cleaned_result.endswith('```'):
            cleaned_result = re.sub(r'\s*```$', '', cleaned_result)
        
        # 解析JSON结果
        chunks_data = json.loads(cleaned_result)
        
        # 处理不同的返回格式
        if "chunks" in chunks_data:
            return chunks_data["chunks"]
        elif "slice" in chunks_data:
            if isinstance(chunks_data, list):
                return [item.get("slice", "") for item in chunks_data if item.get("slice")]
            else:
                return [chunks_data["slice"]]
        else:
            if isinstance(chunks_data, list):
                return chunks_data
            else:
                return []
        
    except Exception as e:
        print(f"LLM切片失败: {e}")
        return semantic_chunking(text, max_chunk_size)

# 4. 滑动窗口切片
def sliding_window_chunking(text, window_size=512, step_size=256):
    """滑动窗口切片"""
    chunks = []
    
    for i in range(0, len(text), step_size):
        chunk = text[i:i + window_size]
        
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())
    
    return chunks

# 5. 自适应切片
def adaptive_chunking(text, target_size=512, tolerance=0.2):
    """自适应切片 - 根据内容自适应调整"""
    chunks = []
    
    # 按段落分割
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # 如果当前段落加入后超过目标大小
        if len(current_chunk) + len(paragraph) > target_size * (1 + tolerance):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph if current_chunk else paragraph
    
    # 处理最后一个块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# 6. 智能自适应切片
def smart_adaptive_chunking(text, target_size=512, min_size=100, max_size=1000):
    """智能自适应切片 - 考虑语义和长度"""
    chunks = []
    
    # 按句子分割
    sentences = re.split(r'[.!?。！？\n]+', text)
    
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 检查是否需要开始新块
        if (len(current_chunk) + len(sentence) > max_size and 
            len(current_chunk) >= min_size):
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        elif len(current_chunk) + len(sentence) > target_size and len(current_chunk) >= min_size:
            # 接近目标长度，考虑是否结束当前块
            if len(sentence) > target_size * 0.3:  # 如果下一句很长，结束当前块
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # 处理最后一个块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def print_chunk_analysis(chunks, method_name):
    """打印切片分析结果"""
    print(f"\n{'='*60}")
    print(f"📋 {method_name}")
    print(f"{'='*60}")
    
    if not chunks:
        print("❌ 未生成任何切片")
        return
    
    total_length = sum(len(chunk) for chunk in chunks)
    avg_length = total_length / len(chunks)
    min_length = min(len(chunk) for chunk in chunks)
    max_length = max(len(chunk) for chunk in chunks)
    
    print(f"📊 统计信息:")
    print(f"   - 切片数量: {len(chunks)}")
    print(f"   - 平均长度: {avg_length:.1f} 字符")
    print(f"   - 最短长度: {min_length} 字符")
    print(f"   - 最长长度: {max_length} 字符")
    print(f"   - 长度方差: {max_length - min_length} 字符")
    
    print(f"\n📝 切片内容:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk}")
        print()

def main():
    """主测试函数"""
    # 测试文本
    text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""
    
    print("🎯 切片策略对比测试")
    print(f"📄 测试文本长度: {len(text)} 字符")
    
    # 测试参数
    target_size = 300
    
    # 1. 固定长度切片
    chunks1 = improved_fixed_length_chunking(text, chunk_size=target_size, overlap=50)
    print_chunk_analysis(chunks1, "1. 固定长度切片")
    
    # 2. 句子边界切片
    chunks2 = semantic_chunking(text, max_chunk_size=target_size)
    print_chunk_analysis(chunks2, "2. 句子边界切片")
    
    # 3. LLM语义切片（LLM）
    print("\n🤖 正在调用LLM进行语义切片...")
    chunks3 = advanced_semantic_chunking_with_llm(text, max_chunk_size=target_size)
    print_chunk_analysis(chunks3, "3. LLM语义切片（LLM）")
    
    # 4. 滑动窗口切片
    chunks4 = sliding_window_chunking(text, window_size=target_size, step_size=target_size//2)
    print_chunk_analysis(chunks4, "4. 滑动窗口切片")
    
    # 5. 自适应切片
    chunks5 = adaptive_chunking(text, target_size=target_size, tolerance=0.3)
    print_chunk_analysis(chunks5, "5. 自适应切片")
    
    # 6. 智能自适应切片
    chunks6 = smart_adaptive_chunking(text, target_size=target_size, min_size=100, max_size=500)
    print_chunk_analysis(chunks6, "6. 智能自适应切片")
    
    # 总结对比
    print(f"\n{'='*80}")
    print("📈 策略对比总结")
    print(f"{'='*80}")
    
    methods = [
        ("固定长度", chunks1),
        ("句子边界切片", chunks2),
        ("LLM语义切片", chunks3),
        ("滑动窗口", chunks4),
        ("自适应切片", chunks5),
        ("智能自适应", chunks6)
    ]
    
    print(f"{'策略':<12} {'切片数':<6} {'平均长度':<8} {'长度方差':<8} {'推荐度':<8}")
    print("-" * 50)
    
    for name, chunks in methods:
        if chunks:
            avg_len = sum(len(c) for c in chunks) / len(chunks)
            min_len = min(len(c) for c in chunks)
            max_len = max(len(c) for c in chunks)
            variance = max_len - min_len
            
            # 简单的推荐度评估
            if len(chunks) >= 2 and variance < 100 and avg_len > 150:
                recommendation = "⭐⭐⭐⭐⭐"
            elif len(chunks) >= 2 and variance < 150:
                recommendation = "⭐⭐⭐⭐"
            elif len(chunks) >= 1:
                recommendation = "⭐⭐⭐"
            else:
                recommendation = "⭐⭐"
            
            print(f"{name:<12} {len(chunks):<6} {avg_len:<8.1f} {variance:<8.1f} {recommendation:<8}")
        else:
            print(f"{name:<12} {'0':<6} {'N/A':<8} {'N/A':<8} {'⭐':<8}")

if __name__ == "__main__":
    import json
    main() 