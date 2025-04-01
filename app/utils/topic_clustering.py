import numpy as np
import requests
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import logging

# 配置jieba日志级别
jieba.setLogLevel(logging.INFO)

class BGETopicClusterer:
    """
    基于BGE(BAAI General Embedding)模型的话题聚类器
    使用Ollama API生成嵌入
    """
    def __init__(self, ollama_url="http://localhost:11434", model_name="bge-m3", num_clusters=10, 
                 min_clusters=2, max_clusters=20, auto_adjust=True, db=None):
        """
        初始化话题聚类器
        
        参数:
        - ollama_url: Ollama API的URL
        - model_name: 模型名称，默认为"bge-m3"
        - num_clusters: 话题聚类数量
        - min_clusters: 最小聚类数
        - max_clusters: 最大聚类数
        - auto_adjust: 是否自动调整聚类数量
        - db: Database对象，用于持久化存储嵌入
        """
        self.embedding_cache = EmbeddingCache()

        self.ollama_url = ollama_url
        self.model_name = model_name
        self.num_clusters = num_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.auto_adjust = auto_adjust
        self.db = db
        
        # 初始化K-means聚类器
        self.kmeans = KMeans(n_clusters=num_clusters)
        
        # 存储数据
        self.documents = []
        self.doc_embeddings = []
        self.doc_clusters = None
        
        # 存储聚类结果
        self.clusters = {}
    
    def _get_embedding_from_ollama(self, text, content_id=None):
        """
        使用Ollama API获取文本嵌入，带缓存
        
        参数:
        - text: 要嵌入的文本
        - content_id: 可选内容ID，用于持久化存储
        """
        # 检查内存缓存
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # 如果提供了内容ID和数据库连接，尝试从数据库获取
        if content_id and self.db:
            stored_embedding = self.db.get_embedding(content_id)
            if stored_embedding is not None:
                # 存入内存缓存
                self.embedding_cache.set(text, stored_embedding)
                return stored_embedding
        
        try:
            # 通过API获取嵌入
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            embedding = np.array(result.get("embedding", []))
            
            # 验证嵌入有效性
            if np.any(embedding) and not np.isnan(embedding).any():
                # 存入内存缓存
                self.embedding_cache.set(text, embedding)
                
                # 如果提供了内容ID和数据库连接，持久化存储
                if content_id and self.db:
                    self.db.save_embedding(content_id, embedding)
                    
                return embedding
            else:
                print(f"警告: 获取到无效嵌入向量")
                return np.zeros(1024)
        except Exception as e:
            print(f"获取嵌入时出错: {e}")
            return np.zeros(1024)
    
    def add_document(self, doc_id, text):
        """
        添加文档到聚类器
        
        参数:
        - doc_id: 文档ID
        - text: 文档文本内容
        
        返回:
        - 文档在集合中的索引
        """
        self.documents.append({
            'id': doc_id,
            'text': text
        })
        
        # 如果需要立即生成嵌入，可以在这里调用
        # 但通常我们推迟到encode_documents中批量处理
        
        return len(self.documents) - 1  # 返回文档索引
    
    # 在BGETopicClusterer类中添加嵌入验证
    def encode_documents(self):
        """为所有文档生成BGE嵌入"""
        self.doc_embeddings = []
        valid_docs = 0
        for doc in self.documents:
            print(f"为文档生成嵌入: {doc['id']}")
            embedding = self._get_embedding_from_ollama(doc['text'])
            
            # 验证嵌入向量有效性
            if np.any(embedding) and not np.isnan(embedding).any():
                self.doc_embeddings.append(embedding)
                valid_docs += 1
            else:
                print(f"警告: 文档 {doc['id']} 生成的嵌入无效，跳过")
        
        print(f"有效文档数: {valid_docs}/{len(self.documents)}")
        return np.array(self.doc_embeddings) if self.doc_embeddings else np.array([])
    
    def _find_optimal_clusters(self, embeddings, min_clusters=2, max_clusters=20):
        """
        使用轮廓系数找到最佳聚类数量
        """
        n_samples = len(embeddings)
        max_k = min(max_clusters, n_samples - 1)
        min_k = min(min_clusters, n_samples - 1)
        
        if min_k < 2:
            return 2  # 至少需要2个聚类
        
        if n_samples <= max_k:
            return min(5, n_samples)  # 如果样本数少，使用较小的聚类数
        
        print(f"寻找最佳聚类数量，范围 {min_k} 到 {max_k}...")
        
        best_score = -1
        best_k = self.num_clusters
        
        # 尝试不同的聚类数量
        for k in range(min_k, max_k + 1):
            # 跳过某些数值以加速处理
            if k > 10 and k % 2 != 0:
                continue
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # 计算轮廓系数
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                print(f"聚类数={k}, 轮廓系数={silhouette_avg:.4f}")
                
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_k = k
            except Exception as e:
                print(f"计算聚类数={k}的轮廓系数出错: {e}")
        
        print(f"最佳聚类数量: {best_k}, 轮廓系数: {best_score:.4f}")
        return best_k
    
    def cluster(self):
        """
        ~~使用K-means聚类文档~~
        使用HDBSCAN聚类文档(03.31更新)
        """
        if not self.doc_embeddings:
            self.encode_documents()
        
        # 确保我们有足够的文档来聚类
        n_samples = len(self.doc_embeddings)
        if n_samples < 2:
            print(f"警告: 文档数量({n_samples})不足以进行聚类")
            self.doc_clusters = np.zeros(n_samples, dtype=int)
            self.clusters = {0: [self.documents[0]]} if n_samples > 0 else {}
            return self.clusters
        
        # 使用HDBSCAN进行聚类
        self.clusters = self.cluster_with_hdbscan()
        
        return self.clusters
    
    def cluster_with_hdbscan(self):
        """使用HDBSCAN聚类文档"""
        if not self.doc_embeddings:
            self.encode_documents()
        
        n_samples = len(self.doc_embeddings)
        if n_samples < 2:
            print(f"警告: 文档数量({n_samples})不足以进行聚类")
            self.doc_clusters = np.zeros(n_samples, dtype=int)
            self.clusters = {0: [self.documents[0]]} if n_samples > 0 else {}
            return self.clusters
        
        # HDBSCAN参数 - 可根据数据特性调整
        min_cluster_size = max(3, int(n_samples * 0.05))  # 至少3个文档或5%的文档
        min_samples = 2  # 更宽松的连接性要求
        
        print(f"执行HDBSCAN聚类, 参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        # 执行聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.5,  # 帮助捕获低密度区域
            cluster_selection_method='eom'  # 使用Excess of Mass方法
        )
        
        self.doc_clusters = clusterer.fit_predict(self.doc_embeddings)
        
        # 整理聚类结果
        self.clusters = {}
        for i, cluster_id in enumerate(self.doc_clusters):
            # HDBSCAN将噪声点标记为-1
            if cluster_id == -1:
                cluster_id = -1  # 保持噪声点标签
                
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(self.documents[i])
        
        # 保存簇概率信息
        self.cluster_probabilities = clusterer.probabilities_
        
        # 打印聚类统计信息
        n_clusters = len(set(self.doc_clusters)) - (1 if -1 in self.doc_clusters else 0)
        n_noise = list(self.doc_clusters).count(-1)
        print(f"HDBSCAN聚类完成: 找到{n_clusters}个聚类, {n_noise}个噪声点")
        
        return self.clusters
    
    def classify_document(self, text):
        """
        将新文档分类到现有聚类中
        """
        # 获取文档嵌入
        embedding = self._get_embedding_from_ollama(text)
        
        # 预测聚类
        cluster_id = self.kmeans.predict([embedding])[0]
        
        # 返回聚类信息
        return {
            'cluster_id': cluster_id,
            'documents': self.clusters.get(cluster_id, []),
            'keywords': self.extract_keywords(cluster_id)
        }
    
    def extract_keywords(self, cluster_id, top_n=5):
        """
        为指定聚类提取关键词（使用jieba提取关键词）
        """
        if cluster_id not in self.clusters:
            return []
        
        # 将聚类中的所有文本合并
        texts = [doc['text'] for doc in self.clusters[cluster_id]]
        combined_text = ' '.join(texts)
        
        # 使用jieba提取关键词
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(combined_text, topK=top_n)
        
        return keywords
    
    def find_similar_documents(self, text, top_n=5):
        """
        查找与给定文本最相似的文档
        """
        if not self.doc_embeddings:
            return []
        
        # 获取查询文本的嵌入
        query_embedding = self._get_embedding_from_ollama(text)
        
        # 计算与所有文档的相似度
        similarities = []
        for i, doc_embedding in enumerate(self.doc_embeddings):
            sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((i, sim))
        
        # 排序并返回最相似的文档
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_docs = []
        for i, sim in similarities[:top_n]:
            top_docs.append({
                'document': self.documents[i],
                'similarity': sim,
                'cluster_id': self.doc_clusters[i] if self.doc_clusters is not None else None
            })
        
        return top_docs
    
class EmbeddingCache:
    """嵌入向量内存缓存"""
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.stats = {"hits": 0, "misses": 0}
        
    def get(self, text):
        """获取缓存中的嵌入向量"""
        key = hash(text)
        embedding = self.cache.get(key)
        if embedding is not None:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        return embedding
        
    def set(self, text, embedding):
        """存储嵌入向量到缓存"""
        key = hash(text)
        
        # 如果缓存满了，删除最早的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[key] = embedding
        
    def get_stats(self):
        """获取缓存统计信息"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }