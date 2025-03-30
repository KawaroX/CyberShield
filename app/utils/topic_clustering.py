import numpy as np
import requests
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
                 min_clusters=2, max_clusters=20, auto_adjust=True):
        """
        初始化话题聚类器
        
        参数:
        - ollama_url: Ollama API的URL
        - model_name: 模型名称，默认为"bge-m3"
        - num_clusters: 话题聚类数量
        - min_clusters: 最小聚类数
        - max_clusters: 最大聚类数
        - auto_adjust: 是否自动调整聚类数量
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.num_clusters = num_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.auto_adjust = auto_adjust
        
        # 初始化K-means聚类器
        self.kmeans = KMeans(n_clusters=num_clusters)
        
        # 存储数据
        self.documents = []
        self.doc_embeddings = []
        self.doc_clusters = None
        
        # 存储聚类结果
        self.clusters = {}
    
    def _get_embedding_from_ollama(self, text):
        """
        使用Ollama API获取文本嵌入
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result.get("embedding", []))
        except Exception as e:
            print(f"获取嵌入时出错: {e}")
            # 返回零向量作为回退
        return np.zeros(1024)  # BGE嵌入通常是1024维
    
    def add_document(self, doc_id, text):
        """
        添加文档到聚类器
        """
        self.documents.append({
            'id': doc_id,
            'text': text
        })
        return len(self.documents) - 1  # 返回文档索引
    
    def encode_documents(self):
        """
        为所有文档生成BGE嵌入
        """
        self.doc_embeddings = []
        for doc in self.documents:
            print(f"为文档生成嵌入: {doc['id']}")
            embedding = self._get_embedding_from_ollama(doc['text'])
            self.doc_embeddings.append(embedding)
        
        return np.array(self.doc_embeddings)
    
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
        使用K-means聚类文档
        """
        if not self.doc_embeddings:
            self.encode_documents()
        
        # 确保我们有足够的文档来聚类
        n_samples = len(self.doc_embeddings)
        if n_samples < 2:
            print(f"警告: 文档数量({n_samples})不足以进行聚类")
            # 如果只有一个文档，直接将其分配到第一个聚类
            self.doc_clusters = np.zeros(n_samples, dtype=int)
            
            # 整理聚类结果
            self.clusters = {0: [self.documents[0]]} if n_samples > 0 else {}
            return self.clusters
        
        # 自动调整聚类数量
        if self.auto_adjust and n_samples >= self.min_clusters:
            optimal_clusters = self._find_optimal_clusters(
                self.doc_embeddings, 
                self.min_clusters, 
                min(self.max_clusters, n_samples - 1)
            )
            
            # 更新聚类数
            self.num_clusters = optimal_clusters
            self.kmeans = KMeans(n_clusters=optimal_clusters)
        else:
            # 使用固定聚类数，但确保不超过样本数
            actual_clusters = min(self.num_clusters, n_samples - 1)
            if actual_clusters != self.num_clusters:
                print(f"调整聚类数 {self.num_clusters} -> {actual_clusters}，匹配样本数量")
                self.kmeans = KMeans(n_clusters=actual_clusters)
        
        # 执行聚类
        self.doc_clusters = self.kmeans.fit_predict(self.doc_embeddings)
        
        # 整理聚类结果
        self.clusters = {}
        for i, cluster_id in enumerate(self.doc_clusters):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(self.documents[i])
        
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