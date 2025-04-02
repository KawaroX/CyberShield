import os

class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cybershield-default-key'
    DEBUG = True
    
    # 百度API配置
    BAIDU_API_KEY = os.environ.get('BAIDU_API_KEY') or 'y2CX2YIdUdDek9ldZq7mbZJm'
    BAIDU_SECRET_KEY = os.environ.get('BAIDU_SECRET_KEY') or 'zlzAL3BObj6QCr2ZBBZ0CRyjjioBc4dv'
    
    # Ollama配置
    OLLAMA_URL = os.environ.get('OLLAMA_URL') or 'http://localhost:11434'
    EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL') or 'bge-m3'

    # MongoDB配置
    MONGODB_URI = os.environ.get('MONGODB_URI') or 'mongodb://localhost:27017/'
    
    # 聚类配置
    NUM_CLUSTERS = int(os.environ.get('NUM_CLUSTERS') or 10)

    # 微调模型路径
    VIOLENCE_MODEL_PATH = os.environ.get('VIOLENCE_MODEL_PATH') or '/Volumes/base/violence_embedding/models/latest'
    # 应用配置
    VIOLENCE_THRESHOLD = 0.7  # 暴力内容阈值
    EARLY_WARNING_THRESHOLD = 0.5  # 预警阈值