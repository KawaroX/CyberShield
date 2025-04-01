from pymongo import MongoClient
import datetime
import numpy as np
from bson import ObjectId

def json_serialize(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        return obj

def convert_mongo_doc(doc):
    """转换MongoDB文档为JSON可序列化格式"""
    if doc is None:
        return None
        
    # 创建新的字典，避免修改原始文档
    result = {}
    for key, value in doc.items():
        if key == '_id':
            # 转换 ObjectId 为字符串
            result['id'] = str(value)
        elif isinstance(value, list):
            # 递归处理列表中的每个元素
            result[key] = [convert_mongo_doc(item) if isinstance(item, dict) else json_serialize(item) for item in value]
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            result[key] = convert_mongo_doc(value)
        else:
            # 其他类型直接序列化
            result[key] = json_serialize(value)
    
    return result

class Database:
    def __init__(self, connection_uri="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_uri)
        self.db = self.client.cybershield
        
        # 创建集合
        self.contents = self.db.contents
        self.topics = self.db.topics
        self.clusters = self.db.clusters
        
        # 创建索引
        self.contents.create_index("content_id")
        self.topics.create_index("topic_id")
        self.contents.create_index("created_at")

    def _load_historical_data(self):
        """加载历史数据到聚类器，用于初始化"""
        contents = self.db.get_recent_contents_for_clustering(days=7, limit=1000)
        print(f"从数据库加载了 {len(contents)} 条历史内容")
        
        # 添加到聚类器
        for content in contents:
            content_id = content["content_id"]
            text = content.get("raw_content", "")
            
            # 尝试获取存储的嵌入向量
            stored_embedding = self.db.get_embedding(content_id)
            
            if stored_embedding is not None:
                # 直接使用存储的嵌入
                self.clusterer.add_document_with_embedding(content_id, text, stored_embedding)
            elif text:
                # 如果没有存储的嵌入，重新计算
                self.clusterer.add_document(content_id, text)
        
    def insert_content(self, content_analysis):
        """插入内容分析结果"""
        # 检查是否为字典或对象
        if isinstance(content_analysis, dict):
            content_dict = content_analysis
        else:
            # 如果是对象，调用to_dict方法
            content_dict = content_analysis.to_dict()
        
        # 添加时间戳
        content_dict["created_at"] = datetime.datetime.now()
        
        # 存储原始内容
        if isinstance(content_analysis, dict):
            if "content" in content_analysis:
                content_dict["raw_content"] = content_analysis["content"]
        else:
            content_dict["raw_content"] = content_analysis.content
        
        # 插入数据库
        result = self.contents.insert_one(content_dict)
        return result.inserted_id
    
    def insert_topic(self, topic_analysis):
        """插入话题分析结果"""
        # 转换为字典
        topic_dict = topic_analysis.to_dict()
        
        # 插入或更新
        result = self.topics.update_one(
            {"topic_id": topic_dict["topic_id"]},
            {"$set": topic_dict},
            upsert=True
        )
        return result
    
    def update_topic(self, topic_id, update_data):
        """更新话题信息"""
        result = self.topics.update_one(
            {"topic_id": topic_id},
            {"$set": update_data}
        )
        return result
    
    def add_content_to_topic(self, topic_id, content_id):
        """将内容关联到话题"""
        result = self.topics.update_one(
            {"topic_id": topic_id},
            {"$addToSet": {"content_ids": content_id}}
        )
        return result
    
    def get_all_contents(self, limit=1000):
        """获取所有内容（用于重新聚类）"""
        contents = list(self.contents.find().sort("created_at", -1).limit(limit))
        return contents
    
    def get_content(self, content_id):
        """获取特定内容"""
        return self.contents.find_one({"content_id": content_id})
    
    def get_topic(self, topic_id):
        """获取特定话题"""
        return self.topics.find_one({"topic_id": topic_id})
    
    def get_all_topics(self):
        """获取所有话题"""
        return list(self.topics.find())
    
    def get_topic_contents(self, topic_id, limit=100):
        """获取话题相关的所有内容"""
        topic = self.topics.find_one({"topic_id": topic_id})
        if not topic or "content_ids" not in topic:
            return []
                
        content_ids = topic["content_ids"]
        contents = list(self.contents.find({"content_id": {"$in": content_ids}}).limit(limit))
        
        # 转换为 JSON 可序列化格式
        return [convert_mongo_doc(content) for content in contents]
    
    def save_embedding(self, content_id, embedding):
        """存储内容的嵌入向量"""
        self.contents.update_one(
            {"content_id": content_id},
            {"$set": {"embedding": embedding.tolist()}}
        )
        
    def get_embedding(self, content_id):
        """获取内容的嵌入向量"""
        content = self.contents.find_one({"content_id": content_id})
        if content and "embedding" in content:
            return np.array(content["embedding"])
        return None

    def save_cluster_results(self, cluster_mapping):
        """保存聚类结果"""
        # 清空旧的聚类结果
        self.clusters.delete_many({})
        
        # 插入新的聚类结果
        cluster_docs = []
        for content_id, cluster_id in cluster_mapping.items():
            # 将NumPy类型转换为Python原生类型
            if hasattr(cluster_id, 'item'):  # 检查是否是NumPy类型
                cluster_id = cluster_id.item()  # 转换为Python原生类型
            
            cluster_docs.append({
                "content_id": content_id,
                "cluster_id": cluster_id,
                "clustered_at": datetime.datetime.now()
            })
                
        if cluster_docs:
            self.clusters.insert_many(cluster_docs)
            
        return len(cluster_docs)
    
    def get_recent_contents_for_clustering(self, days=7, limit=1000):
        """获取最近一段时间的内容用于聚类"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        return list(self.contents.find(
            {"created_at": {"$gte": cutoff_date}}
        ).sort("created_at", -1).limit(limit))
    
    def get_content_topics(self, content_id):
        """获取内容所属的所有话题ID列表"""
        # 查找包含该内容ID的所有话题
        topics = list(self.topics.find({"content_ids": content_id}))
        return [topic["topic_id"] for topic in topics]

    def get_all_topics(self):
        """获取所有话题"""
        topics = list(self.topics.find())
        return [convert_mongo_doc(topic) for topic in topics]

    def search_topics(self, keyword, limit=10):
        """搜索话题"""
        # 基于关键词搜索话题
        regex = {"$regex": keyword, "$options": "i"}
        topics = list(self.topics.find({
            "$or": [
                {"topic_id": regex},
                {"keywords": regex}
            ]
        }).limit(limit))
        return [convert_mongo_doc(topic) for topic in topics]
    
    def json_serialize(obj):
        """将对象转换为JSON可序列化的格式"""
        from bson import ObjectId
        
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        else:
            return obj

    def convert_mongo_doc(doc):
        """转换MongoDB文档为JSON可序列化格式"""
        if doc is None:
            return None
            
        # 创建新的字典，避免修改原始文档
        result = {}
        for key, value in doc.items():
            if key == '_id':
                # 转换 ObjectId 为字符串
                result['id'] = str(value)
            elif isinstance(value, list):
                # 递归处理列表中的每个元素
                result[key] = [convert_mongo_doc(item) if isinstance(item, dict) else json_serialize(item) for item in value]
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                result[key] = convert_mongo_doc(value)
            else:
                # 其他类型直接序列化
                result[key] = json_serialize(value)
        
        return result