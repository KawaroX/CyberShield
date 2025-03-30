from flask import Flask
from flask_cors import CORS
from config import Config
from app.utils.baidu_nlp import BaiduNLP
from app.utils.violence_detector import ViolenceDetector
from app.utils.database import Database
from app.utils.topic_manager import TopicManager


# 全局变量，在应用中共享
baidu_nlp = None
violence_detector = None
db = None
topic_manager = None

def create_app(config_class=Config):
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(config_class)
    
    # 初始化百度NLP客户端
    global baidu_nlp
    baidu_nlp = BaiduNLP(
        app.config['BAIDU_API_KEY'],
        app.config['BAIDU_SECRET_KEY']
    )
    
    # 初始化暴力检测器
    global violence_detector
    violence_detector = ViolenceDetector()
    
    # 初始化数据库
    global db
    db = Database(app.config.get('MONGODB_URI', 'mongodb://localhost:27017/'))
    
    # 初始化话题管理器
    global topic_manager
    topic_manager = TopicManager(
        db=db,
        ollama_url=app.config.get('OLLAMA_URL', 'http://localhost:11434'),
        model_name=app.config.get('EMBEDDINGS_MODEL', 'bge'),
        num_clusters=app.config.get('NUM_CLUSTERS', 5)
    )
    
    # 启动定期聚类
    topic_manager.start_periodic_clustering()
    
    # 注册蓝图
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app