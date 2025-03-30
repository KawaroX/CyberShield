import jieba
import re
import random
import json
import os


class ViolenceDetector:
    """
    网络暴力检测器，用于检测文本中的暴力内容
    目前使用简化的暴力词典和规则，未来可以替换为机器学习模型
    """
    
    def __init__(self, violence_dict_path=None):
        # 如果提供词典路径，则加载词典；否则使用默认词典
        if violence_dict_path and os.path.exists(violence_dict_path):
            with open(violence_dict_path, 'r', encoding='utf-8') as f:
                self.violence_dict = json.load(f)
        else:
            # 默认简化暴力词典
            self.violence_dict = {
                # 侮辱类
                "垃圾": 0.3,
                "废物": 0.4,
                "白痴": 0.5,
                "蠢货": 0.4,
                "傻逼": 0.7,
                "傻子": 0.4,
                "废物": 0.4,
                "猪脑子": 0.5,
                
                # 威胁类
                "威胁": 0.6,
                "打死": 0.7,
                "弄死": 0.7,
                "封杀": 0.5,
                "人肉": 0.8,
                "曝光": 0.4,
                "开盒": 0.8,
                
                # 诽谤类
                "造谣": 0.5,
                "抹黑": 0.5,
                "卖国": 0.7,
                "内鬼": 0.5,
                "勾结": 0.6,
                
                # 歧视类
                "歧视": 0.6,
                "种族": 0.4,
                "低等": 0.6,
                "下贱": 0.7,
                
                # 煽动性
                "举报": 0.3,
                "抵制": 0.3,
                "断交": 0.4,
                "封杀": 0.5,
                "冲": 0.3,
                "喷": 0.3
            }
        
        # 暴力类型对应的关键词
        self.violence_types = {
            "harassment": ["垃圾", "废物", "白痴", "蠢货", "傻逼", "傻子", "猪脑子"],
            "threat": ["威胁", "打死", "弄死", "封杀", "人肉", "曝光", "开盒"],
            "defamation": ["造谣", "抹黑", "卖国", "内鬼", "勾结"],
            "discrimination": ["歧视", "种族", "低等", "下贱"],
            "instigation": ["举报", "抵制", "断交", "封杀", "冲"]
        }
        
        # 初始化jieba分词
        jieba.initialize()

    def detect(self, text):
        """
        检测文本中的暴力内容
        返回暴力分数(0-1)、暴力类型和置信度
        """
        # 分词
        words = jieba.lcut(text)
        print(f"分词结果: {words}")
        
        # 记录匹配到的暴力词及其权重
        matched_words = []
        violence_score = 0.0
        
        # 根据词典计算暴力分数
        # for word in words:
        #     if word in self.violence_dict:
        #         violence_score += self.violence_dict[word]
        #         matched_words.append(word)

        for item in self.violence_dict:
            if item in text:
                violence_score += self.violence_dict[item]
                matched_words.append(item)

        print(f"匹配到的暴力词: {matched_words}, 暴力分数: {violence_score}")
                
        # 将暴力分数归一化到0-1范围
        violence_score /= max(self.violence_dict.values(), default=1.0)
        
        # 限制暴力分数范围为0-1
        violence_score = min(violence_score, 1.0)
        
        # 确定暴力类型
        violence_type = self._determine_violence_type(matched_words)
        
        # 计算置信度（简化版，未来可以替换为机器学习模型的置信度）
        confidence_score = 0.7 + 0.2 * random.random() if violence_score > 0.3 else 0.5 + 0.2 * random.random()
        
        return {
            "violence_score": violence_score,
            "violence_type": violence_type,
            "confidence_score": confidence_score,
            "matched_words": matched_words
        }
    
    def _determine_violence_type(self, matched_words):
        """根据匹配到的暴力词确定暴力类型"""
        if not matched_words:
            return None
            
        # 统计每种暴力类型的匹配词数
        type_counts = {t: 0 for t in self.violence_types}
        
        for word in matched_words:
            for t, keywords in self.violence_types.items():
                if word in keywords:
                    type_counts[t] += 1
        
        # 返回匹配词数最多的暴力类型
        if max(type_counts.values()) > 0:
            return max(type_counts, key=type_counts.get)
        
        return None

    def save_violence_dict(self, file_path):
        """保存暴力词典到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.violence_dict, f, ensure_ascii=False, indent=4)
    
    def update_violence_dict(self, new_dict):
        """更新暴力词典"""
        self.violence_dict.update(new_dict)