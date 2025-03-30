import requests
import json
import time


class BaiduNLP:
    """百度自然语言处理API封装类"""

    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = None
        self.expire_time = 0

    def get_access_token(self):
        """获取access_token"""
        # 检查当前token是否过期
        if self.access_token and time.time() < self.expire_time:
            return self.access_token

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }

        response = requests.post(url, params=params)
        result = response.json()

        if "access_token" in result:
            self.access_token = result["access_token"]
            # 设置过期时间(提前10分钟过期，保险起见)
            self.expire_time = time.time() + result.get("expires_in", 2592000) - 600
            return self.access_token
        else:
            raise Exception(f"获取access_token失败: {result}")

    def sentiment_analyze(self, text):
        """情感倾向分析"""
        url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify"

        # 获取access_token
        access_token = self.get_access_token()

        # 构建请求参数
        params = {"access_token": access_token, "charset": "UTF-8"}
        payload = json.dumps({"text": text}, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # 发送请求
        response = requests.post(url, params=params, data=payload.encode("utf-8"))
        result = response.json()

        if "items" in result and len(result["items"]) > 0:
            print(result)
            # 返回情感分析结果
            return {
                "sentiment": result["items"][0]["sentiment"],
                "confidence": float(result["items"][0]["confidence"]),  # 确保是浮点数
                "positive_prob": float(result["items"][0]["positive_prob"]),  # 确保是浮点数
                "negative_prob": float(result["items"][0]["negative_prob"])  # 确保是浮点数
            }
        else:
            if "error_msg" in result:
                raise Exception(f"情感分析失败: {result['error_msg']}")
            else:
                raise Exception(f"情感分析失败: {result}")