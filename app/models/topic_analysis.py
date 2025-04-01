import datetime

class TopicAnalysis:
    """
    宏观话题分析模型，用于分析话题的发展趋势和风险程度
    """
    def __init__(self, topic_id, keywords=None):
        # 基本属性
        self.topic_id = topic_id  # 话题ID
        self.keywords = keywords or []  # 话题关键词
        
        # 时间属性
        self.start_time = datetime.datetime.now()  # 话题开始时间
        self.last_update = datetime.datetime.now()  # 最后更新时间
        
        # 分析结果
        self.sentiment_score = 0.0  # 情感分数 (-1到1，负数表示负面情绪)
        self.negativity_ratio = 0.0  # 负面内容比例
        self.violence_risk_score = 0.0  # 暴力风险分数 (0-1)
        
        # 统计数据
        self.content_count = 0  # 内容总数
        self.users_involved = 0  # 参与用户数
        self.growth_rate = 0.0  # 增长速率

        # 
        self.hourly_stats = {}  # 小时级统计数据
        self.growth_rate = 0.0  # 增长率
        self.risk_acceleration = 0.0  # 风险加速度
        
        # 干预状态
        self.intervention_status = "Monitoring"  # 干预状态 (Monitoring, EarlyWarning, ActiveIntervention)

    def update_stats(self, content_analysis):
        """基于微观内容分析结果更新话题统计数据"""
        self.content_count += 1
        self.last_update = datetime.datetime.now()
        
        # 更新情感分数和负面比例
        if content_analysis.sentiment is not None:
            if content_analysis.sentiment == 0:  # 消极
                self.negativity_ratio = (self.negativity_ratio * (self.content_count - 1) + 1) / self.content_count
            elif content_analysis.sentiment == 2:  # 积极
                self.negativity_ratio = (self.negativity_ratio * (self.content_count - 1) + 0) / self.content_count
            else:  # 中性
                self.negativity_ratio = (self.negativity_ratio * (self.content_count - 1) + 0.5) / self.content_count
        
        # 更新暴力风险分数 (简单加权平均)
        self.violence_risk_score = (self.violence_risk_score * (self.content_count - 1) + content_analysis.violence_score) / self.content_count
        
        # 更新干预状态
        self._update_intervention_status()

    def _update_intervention_status(self):
        """基于分析结果更新干预状态"""
        if self.violence_risk_score > 0.7:
            self.intervention_status = "ActiveIntervention"
        elif self.violence_risk_score > 0.5:
            self.intervention_status = "EarlyWarning"
        else:
            self.intervention_status = "Monitoring"

    def to_dict(self):
        """将对象转换为字典，用于JSON序列化"""
        return {
            'topic_id': self.topic_id,
            'keywords': self.keywords,
            'start_time': self.start_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'sentiment_score': self.sentiment_score,
            'negativity_ratio': self.negativity_ratio,
            'violence_risk_score': self.violence_risk_score,
            'content_count': self.content_count,
            'users_involved': self.users_involved,
            'growth_rate': self.growth_rate,
            'intervention_status': self.intervention_status
        }
    def from_dict(self, data):
        """从字典加载数据"""
        self.topic_id = data.get("topic_id", self.topic_id)
        self.keywords = data.get("keywords", [])
        
        # 处理日期时间字段
        if "start_time" in data:
            if isinstance(data["start_time"], str):
                self.start_time = datetime.datetime.fromisoformat(data["start_time"])
            else:
                self.start_time = data["start_time"]
                
        if "last_update" in data:
            if isinstance(data["last_update"], str):
                self.last_update = datetime.datetime.fromisoformat(data["last_update"])
            else:
                self.last_update = data["last_update"]
        
        # 加载其他字段
        self.sentiment_score = data.get("sentiment_score", 0.0)
        self.negativity_ratio = data.get("negativity_ratio", 0.0)
        self.violence_risk_score = data.get("violence_risk_score", 0.0)
        self.content_count = data.get("content_count", 0)
        self.users_involved = data.get("users_involved", 0)
        self.growth_rate = data.get("growth_rate", 0.0)
        self.intervention_status = data.get("intervention_status", "Monitoring")
        
        return self
    
if __name__ == "__main__":
    topic_analysis = TopicAnalysis(topic_id=1)
    print(topic_analysis.to_dict())