import uuid
import datetime
import threading
import time
from app.models.topic_analysis import TopicAnalysis
from app.utils.topic_clustering import BGETopicClusterer
from app.models.content_analysis import ContentAnalysis
from app.utils.database import Database

class TopicManager:
    """
    话题管理器，负责管理话题和聚类
    """
    def __init__(self, db, ollama_url="http://localhost:11434", model_name="bge-m3", num_clusters=5):
        """
        初始化话题管理器
        
        参数:
        - db: Database实例
        - ollama_url: Ollama API的URL
        - model_name: 模型名称，默认为"bge-m3"
        - num_clusters: 话题聚类数量
        """
        # 数据库连接
        self.db = db
        
        # 初始化聚类器
        self.clusterer = BGETopicClusterer(
            ollama_url=ollama_url,
            model_name=model_name,
            num_clusters=num_clusters
        )
        
        # 定期聚类的设置
        self.clustering_interval = 3600  # 每小时聚类一次（秒）
        self.is_running = False
        self.clustering_thread = None
        
        # 聚类锁（防止并发聚类）
        self.cluster_lock = threading.Lock()
        
        # 加载历史数据到聚类器(最多1000条)
        self._load_historical_data()
    
    def _load_historical_data(self):
        """加载历史数据到聚类器，用于初始化"""
        contents = self.db.get_recent_contents_for_clustering(days=7, limit=1000)
        print(f"从数据库加载了 {len(contents)} 条历史内容")
        
        # 添加到聚类器
        for content in contents:
            content_id = content["content_id"]
            text = content.get("raw_content", "")
            if text:
                self.clusterer.add_document(content_id, text)
        
        # 如果有足够数据，进行初始聚类
        if len(contents) >= 10:
            self.run_clustering()
    
    def add_content(self, content_analysis, context_ids=None, target_topic_id=None):
        """
        添加内容到管理器
        
        参数:
        - content_analysis: ContentAnalysis对象
        - context_ids: 上下文内容ID列表（可选）
        - target_topic_id: 目标话题ID（可选，如果指定则直接绑定到该话题）
        
        返回:
        - 与内容相关的话题分析
        """
        # 保存内容到数据库
        content_id = content_analysis.content_id
        # 将对象转为字典后保存
        if not isinstance(content_analysis, dict):
            content_dict = content_analysis.to_dict()
            content_dict["created_at"] = datetime.datetime.now()
            content_dict["raw_content"] = content_analysis.content
            self.db.contents.insert_one(content_dict)
        
        self.db.insert_content(content_analysis)

        # 添加到聚类器
        self.clusterer.add_document(content_id, content_analysis.content)
        
        # 如果指定了目标话题，直接绑定到该话题
        if target_topic_id:
            print(f"将内容 {content_id} 绑定到话题 {target_topic_id}")
            return self.bind_content_to_topic(content_id, target_topic_id)
        
        # 如果指定了上下文，考虑上下文进行分类
        if context_ids and len(context_ids) > 0:
            return self.classify_with_context(content_analysis, context_ids)
        
        # 如果有足够的内容，尝试进行实时聚类
        contents_count = len(self.clusterer.documents)
        if contents_count % 10 == 0 and contents_count >= 10:  # 每10个内容聚类一次
            self.run_clustering()
            # 尝试将内容分配到现有聚类
            cluster_topic = self._assign_to_cluster(content_analysis)
            if cluster_topic:
                return cluster_topic
        
        # 没有足够内容进行聚类时，创建或使用临时话题
        temp_topic_id = f"temp-topic-{str(uuid.uuid4())[:8]}"
        
        # 创建临时话题
        topic = TopicAnalysis(temp_topic_id, content_analysis.keywords)
        topic.update_stats(content_analysis)
        
        # 保存话题到数据库
        topic_dict = topic.to_dict()
        self.db.topics.update_one(
            {"topic_id": temp_topic_id},
            {"$set": topic_dict},
            upsert=True
        )
        
        # 关联内容到话题
        self.db.add_content_to_topic(temp_topic_id, content_id)
        
        return topic
    
    def bind_content_to_topic(self, content_id, topic_id):
        """
        将内容直接绑定到指定话题
        
        参数:
        - content_id: 内容ID
        - topic_id: 话题ID
        
        返回:
        - 话题分析对象
        """
        # 获取内容
        content_dict = self.db.get_content(content_id)
        if not content_dict:
            print(f"错误: 未找到内容 {content_id}")
            return None
        
        # 检查话题是否存在，不存在则创建
        topic_dict = self.db.get_topic(topic_id)
        if not topic_dict:
            # 创建新话题
            topic = TopicAnalysis(topic_id, content_dict.get("keywords", []))
            self.db.insert_topic(topic)
        else:
            # 获取现有话题
            topic = TopicAnalysis(topic_id)
            topic.from_dict(topic_dict)
        
        # 将内容转换为ContentAnalysis对象
        content_analysis = ContentAnalysis(content_id, content_dict.get("content_type", "text"))
        content_analysis.from_dict(content_dict)
        
        # 更新话题统计数据
        topic.update_stats(content_analysis)
        
        # 更新数据库
        self.db.update_topic(topic_id, topic.to_dict())
        
        # 关联内容到话题
        self.db.add_content_to_topic(topic_id, content_id)
        
        return topic

    def classify_with_context(self, content_analysis, context_ids):
        """
        基于上下文对内容进行分类
        
        参数:
        - content_analysis: 内容分析对象
        - context_ids: 上下文内容ID列表
        
        返回:
        - 话题分析对象
        """
        # 查找上下文内容的话题
        context_topics = {}
        for ctx_id in context_ids:
            # 获取上下文内容所属的话题
            topics = self.db.get_content_topics(ctx_id)
            for topic_id in topics:
                if topic_id not in context_topics:
                    context_topics[topic_id] = 0
                context_topics[topic_id] += 1
        
        # 如果没有找到上下文话题，使用常规分类
        if not context_topics:
            return self.add_content(content_analysis)
        
        # 找出最常见的上下文话题
        main_topic_id = max(context_topics.items(), key=lambda x: x[1])[0]
        
        # 将内容绑定到该话题
        return self.bind_content_to_topic(content_analysis.content_id, main_topic_id)
    
    def _assign_to_cluster(self, content_analysis):
        """
        将内容分配到现有聚类
        """
        # 如果没有聚类结果，返回None
        if (self.clusterer.doc_clusters is None or 
            len(self.clusterer.doc_clusters) == 0 or 
            len(self.clusterer.documents) < 10):
            return None
            
        # 查找内容在聚类器中的索引
        content_id = content_analysis.content_id
        content_index = None
        
        for i, doc in enumerate(self.clusterer.documents):
            if doc.get('id') == content_id:
                content_index = i
                break
                
        if content_index is None or content_index >= len(self.clusterer.doc_clusters):
            return None
            
        # 获取聚类ID
        cluster_id = int(self.clusterer.doc_clusters[content_index])
        topic_id = f"cluster-{cluster_id}"
        
        # 从数据库获取话题，如果不存在则创建
        topic_dict = self.db.get_topic(topic_id)
        if not topic_dict:
            # 提取聚类关键词
            keywords = self.clusterer.extract_keywords(cluster_id)
            topic = TopicAnalysis(topic_id, keywords)
            topic.update_stats(content_analysis)
            self.db.insert_topic(topic)
        else:
            # 更新现有话题
            topic = TopicAnalysis(topic_id)
            topic.keywords = topic_dict.get("keywords", [])
            topic.start_time = datetime.datetime.fromisoformat(topic_dict.get("start_time"))
            topic.last_update = datetime.datetime.now()
            topic.sentiment_score = topic_dict.get("sentiment_score", 0.0)
            topic.negativity_ratio = topic_dict.get("negativity_ratio", 0.0)
            topic.violence_risk_score = topic_dict.get("violence_risk_score", 0.0)
            topic.content_count = topic_dict.get("content_count", 0) + 1
            topic.users_involved = topic_dict.get("users_involved", 0)
            topic.growth_rate = topic_dict.get("growth_rate", 0.0)
            topic.intervention_status = topic_dict.get("intervention_status", "Monitoring")
            
            # 更新统计数据
            topic.update_stats(content_analysis)
            
            # 更新数据库
            self.db.update_topic(topic_id, topic.to_dict())
        
        # 关联内容到话题
        self.db.add_content_to_topic(topic_id, content_id)
        
        return topic
    
    def run_clustering(self):
        """
        执行聚类操作（线程安全）
        """
        # 如果正在聚类，跳过
        if not self.cluster_lock.acquire(blocking=False):
            return
        
        try:
            print("开始执行话题聚类...")
            # 执行聚类
            clusters = self.clusterer.cluster()
            
            # 重建话题关联
            self._rebuild_topic_relations(clusters)
            
            print(f"聚类完成，识别到 {len(clusters)} 个话题：{clusters}")
        finally:
            self.cluster_lock.release()
    
    def _rebuild_topic_relations(self, clusters):
        """
        基于聚类结果重新构建话题关系
        """
        # 准备聚类映射
        cluster_mapping = {}
        
        # 遍历聚类结果
        for cluster_id, documents in clusters.items():
            # 确保cluster_id是Python原生类型
            if hasattr(cluster_id, 'item'):
                cluster_id = cluster_id.item()

            # 创建话题ID
            topic_id = f"cluster-{cluster_id}"
            
            # 提取关键词
            keywords = self.clusterer.extract_keywords(cluster_id)
            
            # 准备话题数据
            topic_data = {
                "topic_id": topic_id,
                "keywords": keywords,
                "last_update": datetime.datetime.now().isoformat(),
                "content_ids": []
            }
            
            # 收集内容ID
            content_ids = []
            for doc in documents:
                doc_id = doc['id']
                content_ids.append(doc_id)
                cluster_mapping[doc_id] = cluster_id
            
            # 更新话题内容关联
            topic_data["content_ids"] = content_ids
            
            # 获取现有话题或创建新话题
            existing_topic = self.db.get_topic(topic_id)
            if existing_topic:
                # 更新现有话题
                topic_data["start_time"] = existing_topic.get("start_time")
                topic_data["sentiment_score"] = existing_topic.get("sentiment_score", 0.0)
                topic_data["negativity_ratio"] = existing_topic.get("negativity_ratio", 0.0)
                topic_data["violence_risk_score"] = existing_topic.get("violence_risk_score", 0.0)
                topic_data["content_count"] = len(content_ids)
                topic_data["users_involved"] = existing_topic.get("users_involved", 0)
                topic_data["growth_rate"] = existing_topic.get("growth_rate", 0.0)
                topic_data["intervention_status"] = existing_topic.get("intervention_status", "Monitoring")
            else:
                # 创建新话题
                topic_data["start_time"] = datetime.datetime.now().isoformat()
                topic_data["sentiment_score"] = 0.0
                topic_data["negativity_ratio"] = 0.0
                topic_data["violence_risk_score"] = 0.0
                topic_data["content_count"] = len(content_ids)
                topic_data["users_involved"] = 0
                topic_data["growth_rate"] = 0.0
                topic_data["intervention_status"] = "Monitoring"
            
            # 保存话题到数据库
            self.db.topics.update_one(
                {"topic_id": topic_id},
                {"$set": topic_data},
                upsert=True
            )
        
        # 保存聚类结果
        self.db.save_cluster_results(cluster_mapping)
        
        # 重新计算话题统计数据
        self._recalculate_topic_stats()
    
    def _recalculate_topic_stats(self):
        """
        重新计算所有话题的统计数据
        """
        topics = self.db.get_all_topics()
        
        for topic_dict in topics:
            topic_id = topic_dict["topic_id"]
            content_ids = topic_dict.get("content_ids", [])
            
            if not content_ids:
                continue
                
            # 重置统计数据
            violence_scores = []
            negative_counts = 0
            
            # 读取内容数据
            for content_id in content_ids:
                content = self.db.get_content(content_id)
                if content:
                    violence_scores.append(content.get("violence_score", 0.0))
                    if content.get("is_negative", False):
                        negative_counts += 1
            
            # 计算新统计数据
            if violence_scores:
                avg_violence_score = sum(violence_scores) / len(violence_scores)
                negativity_ratio = negative_counts / len(content_ids)
                
                # 更新话题统计
                self.db.update_topic(topic_id, {
                    "violence_risk_score": avg_violence_score,
                    "negativity_ratio": negativity_ratio,
                    "content_count": len(content_ids),
                    "last_update": datetime.datetime.now().isoformat()
                })
                
                # 更新干预状态
                self._update_intervention_status(topic_id, avg_violence_score)
    
    def _update_intervention_status(self, topic_id, violence_risk_score):
        """更新话题的干预状态"""
        status = "Monitoring"
        if violence_risk_score > 0.7:
            status = "ActiveIntervention"
        elif violence_risk_score > 0.5:
            status = "EarlyWarning"
            
        self.db.update_topic(topic_id, {
            "intervention_status": status
        })
    
    def start_periodic_clustering(self):
        """
        启动定期聚类线程
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.clustering_thread = threading.Thread(target=self._periodic_clustering_thread)
        self.clustering_thread.daemon = True
        self.clustering_thread.start()
    
    def stop_periodic_clustering(self):
        """
        停止定期聚类线程
        """
        self.is_running = False
        if self.clustering_thread:
            self.clustering_thread.join(timeout=1.0)
    
    def _periodic_clustering_thread(self):
        """
        定期聚类线程函数
        """
        while self.is_running:
            # 等待间隔时间
            time.sleep(self.clustering_interval)
            
            try:
                # 从数据库重新加载历史数据
                self._load_historical_data()
                
                # 执行聚类
                if len(self.clusterer.documents) >= 10:
                    self.run_clustering()
            except Exception as e:
                print(f"定期聚类出错: {e}")
    
    def find_similar_contents(self, text, top_n=5):
        """
        查找与文本相似的内容
        """
        # 使用聚类器查找类似文档
        similar_docs = self.clusterer.find_similar_documents(text, top_n)
        
        # 转换为内容分析对象
        results = []
        for doc_info in similar_docs:
            doc_id = doc_info['document']['id']
            content = self.db.get_content(doc_id)
            if content:
                results.append({
                    'content': content,
                    'similarity': doc_info['similarity'],
                    'cluster_id': doc_info['cluster_id']
                })
        
        return results
    
    def get_all_topics(self):
        """
        获取所有话题
        """
        return self.db.get_all_topics()
    
# 在TopicManager类中的get_topic_contents方法
    def get_topic_contents(self, topic_id):
        """
        获取话题关联的所有内容
        """
        contents = self.db.get_topic_contents(topic_id)
        
        # 如果数据库层已经处理了序列化，这里可能已经是字典列表
        # 如果不是，需要适当处理
        return contents