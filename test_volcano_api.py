#!/usr/bin/env python3
"""
测试火山引擎API的Stream响应
帮助诊断为什么Stream返回0 chunks
"""

from openai import OpenAI

# 从你的配置中填入
API_KEY = "12790bf9-3c9b-49b1-8a24-72036e014d78"  # 请补全完整API Key
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "ep-20250302101529-w6dmj"

print("=" * 60)
print("火山引擎 ARK API Stream 测试")
print("=" * 60)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

test_message = "你好，请用中文介绍一下北京的交通系统"

print(f"\n[1] 创建客户端...")
print(f"    Base URL: {BASE_URL}")
print(f"    Model: {MODEL}")
print(f"    API Key: {API_KEY[:10]}...")

try:
    print(f"\n[2] 发送请求...")
    print(f"    Message: {test_message}")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": test_message}
        ],
        stream=True
    )
    
    print(f"\n[3] 收到响应")
    print(f"    类型: {type(response)}")
    print(f"    对象: {response}")
    
    print(f"\n[4] 开始迭代Stream...")
    chunk_count = 0
    total_content = ""
    
    for chunk in response:
        chunk_count += 1
        
        # 打印前3个chunk的详细信息
        if chunk_count <= 3:
            print(f"\n--- Chunk {chunk_count} ---")
            print(f"Type: {type(chunk)}")
            print(f"Object: {chunk}")
            
            if hasattr(chunk, '__dict__'):
                print(f"Dict: {chunk.__dict__}")
            
            print(f"Attributes: {[attr for attr in dir(chunk) if not attr.startswith('_')]}")
            
            if hasattr(chunk, 'choices'):
                print(f"Choices: {chunk.choices}")
                if len(chunk.choices) > 0:
                    print(f"First choice: {chunk.choices[0]}")
                    print(f"Delta: {chunk.choices[0].delta}")
                    if hasattr(chunk.choices[0].delta, 'content'):
                        print(f"Content: {repr(chunk.choices[0].delta.content)}")
        
        # 提取内容
        try:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    total_content += delta.content
                    if chunk_count <= 3:
                        print(f"✓ Content extracted: {repr(delta.content)}")
        except Exception as e:
            print(f"✗ Error extracting content: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"[5] Stream迭代完成")
    print(f"    总chunk数: {chunk_count}")
    print(f"    累计内容长度: {len(total_content)}")
    print(f"\n[6] 完整响应:")
    print("-" * 60)
    print(total_content)
    print("-" * 60)
    
    if chunk_count == 0:
        print("\n❌ 警告: Stream返回了0个chunks!")
        print("   可能原因:")
        print("   1. API配额用完")
        print("   2. 模型不可用")
        print("   3. 请求被拒绝")
        print("   4. API格式不兼容")
    elif len(total_content) == 0:
        print(f"\n⚠️  警告: 收到{chunk_count}个chunks但内容为空!")
        print("   请检查chunk结构是否正确")
    else:
        print(f"\n✅ 测试成功! 收到{chunk_count}个chunks, 总计{len(total_content)}字符")

except Exception as e:
    print(f"\n❌ 错误:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
