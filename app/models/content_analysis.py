import datetime

class ContentAnalysis:
    """
    微观内容分析模型，用于分析单条内容的暴力倾向和情感特征
    """
    def __init__(self, content_id, content_type, content=None):
        # 基本属性
        self.content_id = content_id  # 内容ID
        self.content_type = content_type  # 内容类型 (text, image, video等)
        self.content = content  # 原始内容

        # 分析结果
        self.violence_score = 0.0  # 暴力分数 (0-1)
        self.violence_type = None  # 暴力类型 (harassment, threat, discrimination等)
        self.confidence_score = 0.0  # 置信度

        # 情感分析结果 (来自百度API)
        self.sentiment = None  # 情感极性 (0:消极, 1:中性, 2:积极)
        self.positive_prob = None  # 积极概率
        self.negative_prob = None  # 消极概率
        
        # 其他特征
        self.keywords = []  # 关键词
        self.metadata = {}  # 元数据 (可存储任何额外信息)

    def is_violent(self):
        """判断内容是否为暴力内容 (暴力分数 > 0.7)"""
        return self.violence_score > 0.7

    def is_negative(self):
        """判断内容是否为负面内容"""
        if self.sentiment is not None:
            return self.sentiment == 0  # 0表示消极
        # 如果没有情感分析结果，则基于暴力分数判断
        return self.violence_score > 0.5

    def to_dict(self):
        """将对象转换为字典，用于JSON序列化"""
        return {
            'content_id': self.content_id,
            'content_type': self.content_type,
            'violence_score': self.violence_score,
            'violence_type': self.violence_type,
            'confidence_score': self.confidence_score,
            'sentiment': self.sentiment,
            'positive_prob': self.positive_prob,
            'negative_prob': self.negative_prob,
            'keywords': self.keywords,
            'is_violent': self.is_violent(),
            'is_negative': self.is_negative()
        }

    def from_dict(self, data):
        """从字典加载数据"""
        self.content_id = data.get("content_id", self.content_id)
        self.content_type = data.get("content_type", self.content_type)
        self.violence_score = data.get("violence_score", 0.0)
        self.violence_type = data.get("violence_type")
        self.confidence_score = data.get("confidence_score", 0.0)
        self.sentiment = data.get("sentiment")
        self.positive_prob = data.get("positive_prob")
        self.negative_prob = data.get("negative_prob")
        self.keywords = data.get("keywords", [])
        self.content = data.get("raw_content", "")
        return self