from sentence_transformers import SentenceTransformer
import numpy as np
import math
import os

class EnhancedViolenceDetector:
    """
    基于微调嵌入模型的暴力检测器
    完全兼容原有ViolenceDetector的接口
    """
    def __init__(self, basic_detector=None, model_path=None, db=None):
        """初始化检测器"""
        self.basic_detector = basic_detector  # 保留原检测器作为备用
        self.db = db
        
        # 加载微调模型
        print(f"加载暴力检测嵌入模型: {model_path}")
        try:
            self.model = SentenceTransformer(model_path)
            # 创建参考锚点向量
            self.zero_anchors = self.model.encode([
                "这是一个非常友好的评论。",
                "我完全同意你的观点，说得很好。",
                "谢谢分享这个有用的信息。",
                "这是一个非常愉快的对话。"
                "你好呀，我喜欢你，加油哦，天天开心！:)",
                "谢谢你", "真好看", "我喜欢", "你好", "你好呀", "你好呀，我喜欢你", "你好呀，我喜欢你，加油哦，天天开心！:)",
                "加油加油！", "一定可以的",
            ])
            self.zero_anchor = np.mean(self.zero_anchors, axis=0)

            self.one_anchors = self.model.encode([
                "你这个废物滚出去！",
                "我要杀了你全家！",
                "你们这些人都该死！",
                "傻逼操你妈，你妈死了，你死了，你全家死了",
                "sb", "nmsl", "wdnmd", "zz", "nt", "傻x", "操你妈", "操你妈，你妈死了，你死了，你全家死了",
                "你是个傻逼", "你是个sb", "你是个nmsl", "你是个zz", "你是个nt", "你是个傻x", "你是个操你妈", "你是个操你妈，你妈死了，你死了，你全家死了",
                "你是个傻逼，你是个sb，你是个nmsl，你是个zz，你是个nt，你是个傻x，你是个操你妈，你妈死了，你死了，你全家死了",
            ])
            self.one_anchor = np.mean(self.one_anchors, axis=0)

            self.model_loaded = True
            print("嵌入模型加载成功")
        except Exception as e:
            print(f"加载嵌入模型失败: {e}")
            self.model_loaded = False
            
        # 暴力类型映射表
        self.violence_types = {
            "harassment": ["侮辱", "辱骂", "嘲讽", "贬低", "謾骂"],
            "threat": ["威胁", "恐吓", "勒索", "伤害"],
            "defamation": ["诽谤", "抹黑", "造谣", "污蔑"],
            "discrimination": ["歧视", "仇恨", "偏见"],
            "instigation": ["煽动", "挑唆", "怂恿", "挑衅"]
        }

    def _determine_violence_type(self, score):
        """根据暴力分数推断可能的暴力类型"""
        if score < 0.3:
            return None
        elif score < 0.5:
            return "instigation"  # 低分通常是煽动类
        elif score < 0.7:
            return "harassment"   # 中分通常是侮辱类
        else:
            return "threat"       # 高分通常是威胁类

    def detect(self, text):
        """
        检测文本中的暴力内容
        与原ViolenceDetector接口保持完全一致
        """
        if not self.model_loaded and self.basic_detector:
            return self.basic_detector.detect(text)

        try:
            # 获取文本嵌入
            embedding = self.model.encode([text])
            
            # 使用新方法计算暴力分数
            violence_score = self.calculate_violence_score(embedding[0])
            
            # 应用增强的sigmoid函数进一步拉大差距
            violence_score = self.enhanced_sigmoid(violence_score, k=25)
            
            # 确保范围在0-1之间
            violence_score = min(1.0, max(0.0, violence_score))
            
            # 确定暴力类型
            violence_type = self._determine_violence_type(violence_score)
            
            # 置信度与暴力分数相关联
            confidence_score = 0.5 + 0.4 * abs(violence_score - 0.5) * 2
            
            return {
                "violence_score": violence_score,
                "violence_type": violence_type,
                "confidence_score": confidence_score,
                "matched_words": []
            }
        except Exception as e:
            print(f"嵌入模型检测失败: {e}")
            if self.basic_detector:
                return self.basic_detector.detect(text)
            else:
                return {
                    "violence_score": 0.0, 
                    "violence_type": None,
                    "confidence_score": 0.0,
                    "matched_words": []
                }
            
    def calculate_violence_score(self, embedding):
        # 与正面锚点的余弦相似度
        pos_sim = np.dot(embedding, self.zero_anchor.T) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.zero_anchor))
        
        # 与负面锚点的余弦相似度
        neg_sim = np.dot(embedding, self.one_anchor.T) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.one_anchor))
        
        # 更激进的计算方法，强调差异
        ratio = neg_sim / (pos_sim + 0.001)  # 避免除零
        
        # 非线性变换拉大差距
        score = min(1.0, max(0.0, 0.5 * math.pow(ratio, 2)))
        
        # 如果更接近负面锚点，进一步增强分数
        if neg_sim > pos_sim:
            score = 0.5 + 0.5 * score
        else:
            score = 0.5 * score
            
        return float(score)
    
    def test_samples(self, samples):
        """测试一系列样本，打印每个样本的分数"""
        print("\n=== 暴力检测样本测试 ===")
        for text in samples:
            result = self.detect(text)
            print(f"文本: '{text}'")
            print(f"暴力分数: {result['violence_score']:.2f}")
            print(f"暴力类型: {result['violence_type']}")
            print("-" * 50)

    def enhanced_sigmoid(self, x, k=20, shift=0.5):
        """加强版的sigmoid函数，产生更极端的S曲线"""
        # 更陡峭的S曲线，并加强中心点的位移
        raw = 1 / (1 + math.exp(-k * (x - shift)))
        # 进一步拉伸分布
        if raw > 0.5:
            return pow(raw, 1.5)
        else:
            return pow(raw, 0.5)
