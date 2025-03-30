import requests
import json
import time

BASE_URL = 'http://localhost:5001'  # 使用新端口

def test_single_content():
    """测试单条内容分析"""
    print("\n=== 测试单条内容分析 ===")
    url = f"{BASE_URL}/api/analyze"
    payload = {
        "content": "你好",
        "content_type": "text"
    }
    
    response = requests.post(url, json=payload)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"暴力分数: {result['micro_analysis']['violence_score']:.2f}")
        print(f"情感: {result['micro_analysis']['sentiment']}")
        print(f"操作类型: {result['micro_action']['action_type']}")
        return result
    else:
        print(f"错误: {response.text}")
        return None

def test_context_analysis():
    """测试上下文分析"""
    print("\n=== 测试上下文分析 ===")
    url = f"{BASE_URL}/api/analyze_with_context"
    payload = {
        "content": "我觉得你说得对",
        "content_type": "text",
        "context": [
            {
                "content": "这个人就是个垃圾，做的东西毫无价值",
                "content_id": "text-12345",
                "timestamp": "2025-03-30T04:30:00Z"
            }
        ]
    }
    
    response = requests.post(url, json=payload)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"暴力分数: {result['micro_analysis']['violence_score']:.2f}")
        print(f"情感: {result['micro_analysis']['sentiment']}")
        print(f"操作类型: {result['micro_action']['action_type']}")
        
        # 显示上下文影响信息
        print("\n上下文影响:")
        if 'context_influence' in result and result['context_influence'] is not None:
            print(f"  单独分析暴力分数: {result['context_influence']['solo_violence_score']:.2f}")
            print(f"  最终暴力分数: {result['micro_analysis']['violence_score']:.2f}")
            print(f"  影响类型: {result['context_influence'].get('influence_type', 'unknown')}")
        else:
            print("  未提供上下文影响详情")
        
        return result
    else:
        print(f"错误: {response.text}")
        return None

def test_topic_binding():
    """测试话题绑定"""
    print("\n=== 测试话题绑定 ===")
    
    # 先创建一个话题ID
    topic_id = f"test-topic-{int(time.time())}"
    
    url = f"{BASE_URL}/api/analyze"
    payload = {
        "content": "这个新版本的功能完全不符合用户需求",
        "content_type": "text",
        "topic_id": topic_id
    }
    
    response = requests.post(url, json=payload)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"暴力分数: {result['micro_analysis']['violence_score']:.2f}")
        print(f"情感: {result['micro_analysis']['sentiment']}")
        print(f"绑定话题ID: {result['macro_analysis']['topic_id']}")
        
        # 验证话题ID是否正确
        if result['macro_analysis']['topic_id'] == topic_id:
            print("话题绑定成功!")
        else:
            print(f"话题绑定失败! 预期: {topic_id}, 实际: {result['macro_analysis']['topic_id']}")
        
        return result
    else:
        print(f"错误: {response.text}")
        return None

def test_complex_context():
    """测试复杂上下文场景"""
    print("\n=== 测试复杂上下文场景 ===")
    url = f"{BASE_URL}/api/analyze_with_context"
    payload = {
        "content": "哈哈哈，说得好",
        "content_type": "text",
        "context": [
            {
                "content": "这些人整天就知道喷别人，自己有什么本事",
                "content_id": "text-67890",
                "timestamp": "2025-03-30T04:42:00Z"
            },
            {
                "content": "某某明星就是靠资本炒作，完全没有实力",
                "content_id": "text-78901",
                "timestamp": "2025-03-30T04:41:00Z"
            }
        ]
    }
    
    response = requests.post(url, json=payload)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"暴力分数: {result['micro_analysis']['violence_score']:.2f}")
        print(f"情感: {result['micro_analysis']['sentiment']}")
        print(f"操作类型: {result['micro_action']['action_type']}")
        
        # 显示上下文影响信息
        if 'context_influence' in result:
            print("\n上下文影响:")
            for key, value in result['context_influence'].items():
                print(f"  {key}: {value}")
        
        return result
    else:
        print(f"错误: {response.text}")
        return None

if __name__ == "__main__":
    # 执行测试
    test_single_content()
    time.sleep(1)  # 避免请求过快
    
    test_context_analysis()
    time.sleep(1)
    
    test_topic_binding()
    time.sleep(1)
    
    test_complex_context()