from sentence_transformers import SentenceTransformer
import numpy as np
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
            self.zero_anchor = self.model.encode(["这是一个完全正常的评论，没有任何暴力内容。"])
            self.one_anchor = self.model.encode(["这是极度暴力的辱骂和威胁内容，应该立即删除。"])
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
        # 如果模型未加载成功，回退到基础检测器
        if not self.model_loaded and self.basic_detector:
            print("使用基础检测器回退")
            return self.basic_detector.detect(text)

        try:
            # 获取文本嵌入
            embedding = self.model.encode([text])
            
            # 计算与锚点的相似度
            zero_similarity = np.dot(embedding, self.zero_anchor.T) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.zero_anchor))
            one_similarity = np.dot(embedding, self.one_anchor.T) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.one_anchor))
            
            # 计算暴力分数 (归一化为0-1)
            violence_score = float((one_similarity - zero_similarity + 1) / 2)
            violence_score = min(1.0, max(0.0, violence_score))
            
            # 确定暴力类型
            violence_type = self._determine_violence_type(violence_score)
            
            # 确定置信度 (基于嵌入向量范数)
            confidence_score = min(0.9, max(0.5, float(np.linalg.norm(embedding) / 30.0)))
            
            # 保持与原接口一致的返回格式
            return {
                "violence_score": violence_score,
                "violence_type": violence_type,
                "confidence_score": confidence_score,
                "matched_words": []  # 保持接口一致，但词典模型不再使用
            }
        except Exception as e:
            print(f"嵌入模型检测失败: {e}")
            # 发生错误时回退到基础检测器
            if self.basic_detector:
                return self.basic_detector.detect(text)
            else:
                # 无基础检测器时返回默认结果
                return {
                    "violence_score": 0.0,
                    "violence_type": None,
                    "confidence_score": 0.0,
                    "matched_words": []
                }