# app/utils/trend_analyzer.py
import datetime
import numpy as np
from bson.objectid import ObjectId

class TrendAnalyzer:
    def __init__(self, db):
        self.db = db
        
    def analyze_topic_trends(self, time_window=24):
        """分析所有话题的趋势，返回各个话题的风险变化"""
        topics = self.db.get_all_topics()
        results = {}
        
        now = datetime.datetime.now()
        start_time = now - datetime.timedelta(hours=time_window)
        
        for topic in topics:
            topic_id = topic.get('topic_id')
            if not topic_id:
                continue
                
            topic_trend = self._analyze_single_topic(topic_id, start_time, now)
            results[topic_id] = topic_trend
            
        return results
    
    def _analyze_single_topic(self, topic_id, start_time, end_time):
        """分析单个话题的趋势"""
        # 按小时分组
        hourly_windows = []
        current = start_time
        
        while current < end_time:
            next_hour = current + datetime.timedelta(hours=1)
            hourly_windows.append((current, next_hour))
            current = next_hour
        
        # 收集每小时的数据
        hourly_stats = []
        for start, end in hourly_windows:
            # 查询这个时间段内的内容
            pipeline = [
                {
                    "$match": {
                        "created_at": {"$gte": start, "$lt": end},
                        "topic_id": topic_id
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "count": {"$sum": 1},
                        "avg_violence": {"$avg": "$violence_score"},
                        "max_violence": {"$max": "$violence_score"}
                    }
                }
            ]
            
            result = list(self.db.contents.aggregate(pipeline))
            
            if result and len(result) > 0:
                stats = result[0]
                hourly_stats.append({
                    "hour": start.isoformat(),
                    "count": stats.get("count", 0),
                    "avg_violence": stats.get("avg_violence", 0),
                    "max_violence": stats.get("max_violence", 0)
                })
            else:
                hourly_stats.append({
                    "hour": start.isoformat(),
                    "count": 0,
                    "avg_violence": 0,
                    "max_violence": 0
                })
        
        # 计算趋势指标
        counts = [h["count"] for h in hourly_stats]
        violence_scores = [h["avg_violence"] for h in hourly_stats]
        
        # 分析最近6小时与前6小时的对比
        recent_hours = 6
        if len(counts) >= recent_hours * 2:
            # 讨论量变化
            recent_count = sum(counts[-recent_hours:])
            previous_count = sum(counts[-(recent_hours*2):-recent_hours])
            count_growth = ((recent_count - previous_count) / (previous_count + 1)) * 100  # 百分比变化
            
            # 暴力程度变化
            recent_scores = [s for s, c in zip(violence_scores[-recent_hours:], counts[-recent_hours:]) if c > 0]
            previous_scores = [s for s, c in zip(violence_scores[-(recent_hours*2):-recent_hours], counts[-(recent_hours*2):-recent_hours]) if c > 0]
            
            recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            previous_avg = sum(previous_scores) / len(previous_scores) if previous_scores else 0
            
            violence_change = recent_avg - previous_avg
        else:
            count_growth = 0
            violence_change = 0
        
        # 判断是否需要关注
        needs_attention = (count_growth > 50 and violence_change > 0.1) or (violence_change > 0.3)
        
        return {
            "topic_id": topic_id,
            "hourly_stats": hourly_stats,
            "count_growth_pct": count_growth,
            "violence_change": violence_change,
            "needs_attention": needs_attention
        }
    
    def detect_emerging_risks(self):
        """检测新兴风险话题"""
        # 分析所有话题的趋势
        all_trends = self.analyze_topic_trends(time_window=12)  # 使用较短的时间窗口
        
        # 筛选需要关注的话题
        emerging_risks = []
        for topic_id, trend in all_trends.items():
            if trend["needs_attention"]:
                # 获取话题详情
                topic = self.db.get_topic(topic_id)
                if not topic:
                    continue
                    
                # 添加到风险列表
                emerging_risks.append({
                    "topic_id": topic_id,
                    "keywords": topic.get("keywords", []),
                    "count_growth_pct": trend["count_growth_pct"],
                    "violence_change": trend["violence_change"],
                    "current_risk_score": topic.get("violence_risk_score", 0),
                    "content_count": topic.get("content_count", 0)
                })
        
        # 按风险程度排序
        emerging_risks.sort(key=lambda x: (x["violence_change"], x["count_growth_pct"]), reverse=True)
        
        return emerging_risks