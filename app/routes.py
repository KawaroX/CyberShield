from flask import Blueprint, render_template, jsonify, request
from app import baidu_nlp, violence_detector, topic_manager, db  # 添加了 db
from app.models.content_analysis import ContentAnalysis
from app.models.topic_analysis import TopicAnalysis
from bson import ObjectId
import uuid
import datetime

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """渲染首页"""
    return render_template('index.html')

@main_bp.route('/api/test', methods=['GET', 'POST'])
def test_api():
    """
    简单的测试接口
    """
    return jsonify({
        'status': 'success',
        'message': 'API正常工作',
        'method': request.method,
    })


@main_bp.route('/api/analyze', methods=['POST'])
def analyze_content():
    """
    分析内容接口
    接收JSON格式的请求，包含content字段
    返回分析结果
    """
    data = request.json

    if not data or 'content' not in data:
        return jsonify({'error': '缺少content字段'}), 400

    content = data['content']
    content_type = data.get('content_type', 'text')
    target_topic_id = data.get('topic_id')
    
    if content_type == 'text':
        # 生成内容ID
        content_id = f"text-{str(uuid.uuid4())[:8]}"
        
        # 创建内容分析对象
        content_analysis = ContentAnalysis(content_id, content_type, content)
        
        # 尝试调用百度API进行情感分析
        try:
            sentiment_result = baidu_nlp.sentiment_analyze(content)
            
            # 设置情感分析结果
            content_analysis.sentiment = sentiment_result["sentiment"]
            content_analysis.positive_prob = sentiment_result["positive_prob"]
            content_analysis.negative_prob = sentiment_result["negative_prob"]
            
            # 提取关键词
            try:
                keyword_items = baidu_nlp.keyword_extract(content, 5)
                content_analysis.keywords = [item["word"] for item in keyword_items]
            except Exception as e:
                print(f"百度关键词提取失败: {e}")
                # 使用简单分词作为回退
                content_analysis.keywords = [w for w in content.split() if len(w) > 1][:5]
                
        except Exception as e:
            print(f"百度情感分析API调用失败: {e}")
            # 使用本地简单计算作为回退
            content_analysis.sentiment = 1  # 默认中性
            content_analysis.positive_prob = 0.5
            content_analysis.negative_prob = 0.5
            
            # 简单负面检测
            if any(word in content.lower() for word in ["讨厌", "恨", "烦", "滚", "垃圾"]):
                content_analysis.sentiment = 0  # 消极
                content_analysis.positive_prob = 0.2
                content_analysis.negative_prob = 0.8
            
            # 简单分词作为关键词
            content_analysis.keywords = [w for w in content.split() if len(w) > 1][:5]
        
        # 使用暴力检测器分析内容
        violence_result = violence_detector.detect(content)
        
        # 设置暴力分析结果
        content_analysis.violence_score = violence_result["violence_score"]
        content_analysis.violence_type = violence_result["violence_type"]
        content_analysis.confidence_score = violence_result["confidence_score"]
        
        # 添加内容到话题管理器并获取关联话题
        if target_topic_id:
            topic_analysis = topic_manager.add_content(content_analysis, context_ids=None, target_topic_id=target_topic_id)
        else:
            topic_analysis = topic_manager.add_content(content_analysis)
            
        # 确定微观行动（针对单条内容的行动）
        micro_action = determine_action(content_analysis)
        
        # 生成宏观干预建议（针对话题的干预）
        macro_interventions = generate_interventions(topic_analysis)
        
        # 构建返回结果
        result = {
            'micro_analysis': content_analysis.to_dict(),
            'micro_action': micro_action,
            'macro_analysis': topic_analysis.to_dict(),
            'macro_interventions': macro_interventions
        }
        
        return jsonify(result)
    else:
        return jsonify({'error': f'不支持的内容类型: {content_type}'}), 400


@main_bp.route('/api/topics', methods=['GET'])
def get_topics():
    """
    获取所有话题
    """
    topics = topic_manager.get_all_topics()
    
    # 如果 topics 是列表，直接使用
    # 如果 topics 是字典，转换为列表
    if isinstance(topics, dict):
        topics_list = [topic.to_dict() for topic in topics.values()]
    else:
        # 假设每个元素都是可以转换为字典的对象
        topics_list = [topic if isinstance(topic, dict) else topic.to_dict() for topic in topics]
    
    return jsonify({
        'topics': topics_list
    })


@main_bp.route('/api/topics/<topic_id>/contents', methods=['GET'])
def get_topic_contents(topic_id):
    """
    获取话题关联的所有内容
    """
    contents = topic_manager.get_topic_contents(topic_id)
    
    # 创建一个函数处理单个内容项
    def process_content(content):
        if isinstance(content, dict):
            # 处理字典中可能存在的 ObjectId
            result = {}
            for key, value in content.items():
                if key == '_id':
                    result['id'] = str(value)
                elif isinstance(value, ObjectId):
                    result[key] = str(value)
                else:
                    result[key] = value
            return result
        else:
            # 假设是 ContentAnalysis 对象
            return content.to_dict()
    
    # 处理每个内容项
    formatted_contents = [process_content(content) for content in contents]
    
    return jsonify({
        'contents': formatted_contents
    })


@main_bp.route('/api/similar', methods=['POST'])
def find_similar_contents():
    """
    查找与文本相似的内容
    """
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400
    
    text = data['text']
    top_n = data.get('top_n', 5)
    
    similar_contents = topic_manager.find_similar_contents(text, top_n)
    return jsonify({
        'similar_contents': [
            {
                'content': item['content'].to_dict(),
                'similarity': item['similarity'],
                'cluster_id': item['cluster_id']
            } for item in similar_contents
        ]
    })


@main_bp.route('/api/analyze_with_context', methods=['POST'])
def analyze_with_context():
    """
    分析内容，同时考虑上下文
    """
    data = request.json
    if not data or 'content' not in data:
        return jsonify({'error': '缺少content字段'}), 400

    content = data['content']
    content_type = data.get('content_type', 'text')
    context = data.get('context', [])  # 新的上下文格式
    topic_id = data.get('topic_id')    # 可选话题ID
    
    # 生成内容ID
    content_id = f"text-{str(uuid.uuid4())[:8]}"
    
    # 创建内容分析对象
    content_analysis = ContentAnalysis(content_id, content_type, content)
    
    # 尝试调用百度API进行情感分析
    try:
        sentiment_result = baidu_nlp.sentiment_analyze(content)
        
        # 设置情感分析结果
        content_analysis.sentiment = sentiment_result["sentiment"]
        content_analysis.positive_prob = sentiment_result["positive_prob"]
        content_analysis.negative_prob = sentiment_result["negative_prob"]
        
        # 提取关键词
        try:
            keyword_items = baidu_nlp.keyword_extract(content, 5)
            content_analysis.keywords = [item["word"] for item in keyword_items]
        except Exception as e:
            print(f"百度关键词提取失败: {e}")
            # 使用简单分词作为回退
            content_analysis.keywords = [w for w in content.split() if len(w) > 1][:5]
            
    except Exception as e:
        print(f"百度情感分析API调用失败: {e}")
        # 使用本地简单计算作为回退
        content_analysis.sentiment = 1  # 默认中性
        content_analysis.positive_prob = 0.5
        content_analysis.negative_prob = 0.5
        
        # 简单负面检测
        if any(word in content.lower() for word in ["讨厌", "恨", "烦", "滚", "垃圾"]):
            content_analysis.sentiment = 0  # 消极
            content_analysis.positive_prob = 0.2
            content_analysis.negative_prob = 0.8
        
        # 简单分词作为关键词
        content_analysis.keywords = [w for w in content.split() if len(w) > 1][:5]
    
    # 使用暴力检测器分析内容
    violence_result = violence_detector.detect(content)
    
    # 设置暴力分析结果
    content_analysis.violence_score = violence_result["violence_score"]
    content_analysis.violence_type = violence_result["violence_type"]
    content_analysis.confidence_score = violence_result["confidence_score"]
    
    # 分析上下文内容 - 将已分析的content_analysis传入而不是原始内容
    context_influence = None
    if context:
        print(f"开始分析包含上下文的内容...")
        # 使用智能上下文分析，传入已分析的内容对象
        context_result = analyze_with_context_content(content_analysis, context)
        
        if context_result:
            print(f"更新分析结果: sentiment从{content_analysis.sentiment}到{context_result.sentiment}")
            content_analysis = context_result
            
            # 保存上下文影响数据
            if hasattr(context_result, 'context_influence'):
                context_influence = context_result.context_influence

        # 更新分析结果
        if isinstance(context_result, dict):
            # 如果是字典，更新属性
            for key, value in context_result.items():
                if key != 'to_dict':  # 避免覆盖方法
                    setattr(content_analysis, key, value)

            # 确保context_influence存在
            if 'context_influence' in context_result:
                context_influence = context_result['context_influence']
        else:
            # 复制属性而不是替换对象
            content_analysis.violence_score = context_result.violence_score
            content_analysis.violence_type = context_result.violence_type
            content_analysis.sentiment = context_result.sentiment
            content_analysis.positive_prob = context_result.positive_prob
            content_analysis.negative_prob = context_result.negative_prob
            
            # 保存上下文影响数据
            if hasattr(context_result, 'context_influence'):
                content_analysis.context_influence = context_result.context_influence
    
    # 保存内容到数据库
    db.insert_content(content_analysis.to_dict())
    
    # 将内容添加到话题
    if topic_id:
        topic_analysis = topic_manager.add_content_to_topic(content_analysis, topic_id)
    else:
        topic_analysis = topic_manager.add_content(content_analysis)
    
    # 确定微观行动和生成干预建议
    micro_action = determine_action(content_analysis)
    macro_interventions = generate_interventions(topic_analysis)
    
    # 构建结果
    result = {
        'micro_analysis': content_analysis.to_dict(),
        'micro_action': micro_action,
        'macro_analysis': topic_analysis.to_dict(),
        'macro_interventions': macro_interventions,
        'context_analyzed': bool(context),
        'context_influence': context_influence if context_influence is not None else {
            'solo_violence_score': content_analysis.violence_score,
            'context_violence_score': 0.0,
            'full_violence_score': content_analysis.violence_score,
            'solo_sentiment': content_analysis.sentiment,
            'context_sentiment': None,
            'influence_type': 'none'
        }
    }
    
    return jsonify(result)

@main_bp.route('/api/bind_topic', methods=['POST'])
def bind_topic():
    """
    将内容绑定到指定话题
    """
    data = request.json
    if not data or 'content_id' not in data or 'topic_id' not in data:
        return jsonify({'error': '缺少必要字段'}), 400

    content_id = data['content_id']
    topic_id = data['topic_id']
    
    # 执行绑定
    topic = topic_manager.bind_content_to_topic(content_id, topic_id)
    
    if not topic:
        return jsonify({'error': '绑定失败，可能是内容或话题不存在'}), 404
    
    return jsonify({
        'success': True,
        'topic': topic.to_dict()
    })

@main_bp.route('/api/topics/search', methods=['GET'])
def search_topics():
    """
    搜索话题
    """
    keyword = request.args.get('keyword', '')
    limit = int(request.args.get('limit', 10))
    
    topics = db.search_topics(keyword, limit)
    # topics 现在应该已经被转换为 JSON 可序列化格式
    
    return jsonify({
        'topics': topics
    })


@main_bp.route('/api/analyze_topic', methods=['POST'])
def analyze_topic():
    """
    分析特定话题的聚合情况
    """
    data = request.json
    if not data or 'topic_id' not in data:
        return jsonify({'error': '缺少topic_id字段'}), 400
    
    topic_id = data['topic_id']
    time_range = data.get('time_range', {})
    
    # 获取话题信息
    topic = topic_manager.get_topic(topic_id)
    if not topic:
        return jsonify({'error': f'未找到话题: {topic_id}'}), 404
    
    # 获取话题内容
    contents = topic_manager.get_topic_contents(
        topic_id, 
        start_time=time_range.get('start'),
        end_time=time_range.get('end')
    )
    
    # 计算聚合指标
    total_contents = len(contents)
    violent_contents = sum(1 for c in contents if c.get('violence_score', 0) > 0.7)
    negative_contents = sum(1 for c in contents if c.get('is_negative', False))
    
    # 计算风险分数
    violence_risk = violent_contents / total_contents if total_contents > 0 else 0
    negativity_ratio = negative_contents / total_contents if total_contents > 0 else 0
    
    # 分析结果
    result = {
        'topic_id': topic_id,
        'total_contents': total_contents,
        'violent_contents': violent_contents,
        'negative_contents': negative_contents,
        'violence_risk': violence_risk,
        'negativity_ratio': negativity_ratio,
        'keywords': topic.get('keywords', []),
        'intervention_status': determine_action(violence_risk)
    }
    
    # 生成干预建议
    interventions = generate_interventions(result)
    
    return jsonify({
        'topic_analysis': result,
        'interventions': interventions
    })


def determine_action(analysis):
    """
    确定内容的处理行动
    基于暴力分数和类型确定行动类型、严重程度和提示信息
    """
    # 使用analysis.negative_prob（如果有）和violence_score计算最终分数
    final_score = analysis.violence_score
    if hasattr(analysis, 'negative_prob') and analysis.negative_prob is not None:
        final_score = (final_score + analysis.negative_prob) / 2
    
    # 根据最终分数和暴力类型确定行动
    if final_score > 0.8 and analysis.violence_type == "harassment":
        return {
            "action_type": "remove",
            "severity": "critical",
            "message": "该内容包含严重骚扰行为，已被系统移除",
            "automated": True
        }
    elif final_score > 0.8 or analysis.violence_type == "threat":
        return {
            "action_type": "restrict",
            "severity": "critical",
            "message": "该内容包含威胁或高风险内容，已被限制访问",
            "automated": True
        }
    elif final_score > 0.7:
        return {
            "action_type": "restrict",
            "severity": "high",
            "message": "该内容可能包含有害信息，已被限制可见性",
            "automated": True
        }
    elif final_score > 0.5:
        return {
            "action_type": "warning",
            "severity": "medium",
            "message": "请注意您的言论，避免伤害他人",
            "automated": False
        }
    elif final_score > 0.3:
        return {
            "action_type": "flag",
            "severity": "low",
            "message": "该内容已被标记进行进一步审核",
            "automated": False
        }
    else:
        return {
            "action_type": "none",
            "severity": "none",
            "message": None,
            "automated": False
        }


def generate_interventions(topic_analysis):
    """
    生成话题干预建议
    基于话题分析结果，生成一系列干预策略
    """
    interventions = []
    
    # 基于风险分数生成干预建议
    if topic_analysis.violence_risk_score > 0.7:
        interventions.append({
            "strategy_type": "ContentFiltering",
            "priority": "high",
            "target_audience": "AllUsers",
            "description": "为该话题启用最严格内容过滤，阻止相关内容传播",
            "estimated_impact": 0.9
        })
        interventions.append({
            "strategy_type": "ModeratorAssignment",
            "priority": "high",
            "target_audience": "AllUsers",
            "description": "立即分配专职版主监控该话题，进行人工审核",
            "estimated_impact": 0.8
        })
    elif topic_analysis.violence_risk_score > 0.5:
        interventions.append({
            "strategy_type": "ContentFiltering",
            "priority": "medium",
            "target_audience": "AllUsers",
            "description": "为该话题启用严格内容过滤，限制可见性",
            "estimated_impact": 0.7
        })
        interventions.append({
            "strategy_type": "UserWarning",
            "priority": "medium",
            "target_audience": "ActiveParticipants",
            "description": "向参与讨论的用户发送提醒，避免情绪激化",
            "estimated_impact": 0.6
        })
    elif topic_analysis.violence_risk_score > 0.3:
        interventions.append({
            "strategy_type": "PositiveContentPromotion",
            "priority": "low",
            "target_audience": "AllUsers",
            "description": "在该话题中推广积极内容，平衡讨论氛围",
            "estimated_impact": 0.5
        })
        interventions.append({
            "strategy_type": "AlgorithmAdjustment",
            "priority": "low",
            "target_audience": "AllUsers",
            "description": "适当调整内容推荐算法，减少负面内容曝光",
            "estimated_impact": 0.4
        })
    else:
        interventions.append({
            "strategy_type": "Monitoring",
            "priority": "low",
            "target_audience": "SystemOnly",
            "description": "加强对该话题的监控，定期评估风险变化",
            "estimated_impact": 0.3
        })
    
    return interventions

# 在app/routes.py中修改analyze_with_context_content函数

def analyze_with_context_content(content_analysis, context_list):
    """
    分析内容及其上下文，智能判断上下文影响
    
    参数:
    - content_analysis: 已分析的内容对象
    - context_list: 上下文内容列表，每项包含 'content' 字段
    
    返回:
    - 合并分析结果和上下文影响评估
    """
    # 首先，保存原始分析结果
    solo_analysis = content_analysis
    content = content_analysis.content
    
    # 记录原始值，用于调试和比较
    original_sentiment = solo_analysis.sentiment
    original_violence_score = solo_analysis.violence_score
    
    print(f"原始分析结果: sentiment={original_sentiment}, violence_score={original_violence_score}")
    
    # 获取上下文内容
    context_texts = [ctx.get('content', '') for ctx in context_list]
    combined_context = " \n ".join(context_texts)
    
    # 分析上下文
    temp_id_ctx = f"context-{str(uuid.uuid4())[:8]}"
    context_analysis = ContentAnalysis(temp_id_ctx, "text", combined_context)
    
    # 分析上下文
    if combined_context:
        try:
            # 情感分析
            sentiment_result = baidu_nlp.sentiment_analyze(combined_context)
            context_analysis.sentiment = sentiment_result["sentiment"]
            context_analysis.positive_prob = sentiment_result["positive_prob"]
            context_analysis.negative_prob = sentiment_result["negative_prob"]
            print(f"上下文分析结果: sentiment={context_analysis.sentiment}")
        except Exception as e:
            print(f"上下文情感分析失败: {e}")
            context_analysis.sentiment = 1  # 默认中性
        
        # 暴力检测
        try:
            violence_result = violence_detector.detect(combined_context)
            context_analysis.violence_score = violence_result["violence_score"]
            context_analysis.violence_type = violence_result["violence_type"]
            context_analysis.confidence_score = violence_result["confidence_score"]
            print(f"上下文暴力分析结果: score={context_analysis.violence_score}")
        except Exception as e:
            print(f"上下文暴力检测失败: {e}")
            context_analysis.violence_score = 0.0
    
    # 分析合并内容
    full_text = combined_context + " \n " + content
    
    temp_id_full = f"full-{str(uuid.uuid4())[:8]}"
    full_analysis = ContentAnalysis(temp_id_full, "text", full_text)
    
    # 分析完整内容
    if full_text:
        try:
            # 情感分析
            sentiment_result = baidu_nlp.sentiment_analyze(full_text)
            full_analysis.sentiment = sentiment_result["sentiment"]
            full_analysis.positive_prob = sentiment_result["positive_prob"]
            full_analysis.negative_prob = sentiment_result["negative_prob"]
            print(f"完整内容分析结果: sentiment={full_analysis.sentiment}")
        except Exception as e:
            print(f"完整内容情感分析失败: {e}")
            full_analysis.sentiment = 1  # 默认中性
        
        # 暴力检测
        try:
            violence_result = violence_detector.detect(full_text)
            full_analysis.violence_score = violence_result["violence_score"]
            full_analysis.violence_type = violence_result["violence_type"]
            full_analysis.confidence_score = violence_result["confidence_score"]
            print(f"完整内容暴力分析结果: score={full_analysis.violence_score}")
        except Exception as e:
            print(f"完整内容暴力检测失败: {e}")
            full_analysis.violence_score = 0.0
    
    # 创建结果分析对象
    result_analysis = ContentAnalysis(solo_analysis.content_id, solo_analysis.content_type, content)
    result_analysis.keywords = solo_analysis.keywords
    
    # 初始化影响类型
    influence_type = "none"
    
    # 情感分析逻辑调整 - 保留原始情感类型，除非有明确证据表明应该改变
    # 如果原始内容是积极的(2)，但在负面上下文中出现，整体分析是负面的(0)
    if original_sentiment == 2 and context_analysis.sentiment == 0 and full_analysis.sentiment == 0:
        result_analysis.sentiment = 0  # 改为负面
        result_analysis.positive_prob = 0.3
        result_analysis.negative_prob = 0.7
        influence_type = "positive_to_negative"  # 积极转为负面
    # 如果原始内容是中性的(1)，但在负面上下文中出现，整体分析是负面的(0)
    elif original_sentiment == 1 and context_analysis.sentiment == 0 and full_analysis.sentiment == 0:
        result_analysis.sentiment = 0  # 改为负面
        result_analysis.positive_prob = 0.3
        result_analysis.negative_prob = 0.7
        influence_type = "neutral_to_negative"  # 中性转为负面
    else:
        # 保持原始情感分析结果
        result_analysis.sentiment = original_sentiment
        result_analysis.positive_prob = solo_analysis.positive_prob
        result_analysis.negative_prob = solo_analysis.negative_prob
        influence_type = "unchanged"
    
    # 暴力分数逻辑调整 - 保留原始暴力分数，除非有明确证据表明应该改变
    if original_violence_score < 0.3:  # 当前内容暴力分数低
        if context_analysis.violence_score > 0.3 and full_analysis.violence_score > 0.3:
            # 虽然原内容暴力分数低，但上下文暴力分数高，整体分析也显示暴力
            result_analysis.violence_score = (original_violence_score * 0.3 + 
                                             context_analysis.violence_score * 0.4 + 
                                             full_analysis.violence_score * 0.3)
            influence_type = "amplified_violence"  # 暴力性被放大
            
            # 如果上下文有明确暴力类型，但当前内容没有，则采用上下文的暴力类型
            if not solo_analysis.violence_type and context_analysis.violence_type:
                result_analysis.violence_type = context_analysis.violence_type
        else:
            # 保持原始暴力分数
            result_analysis.violence_score = original_violence_score
            result_analysis.violence_type = solo_analysis.violence_type
    else:
        # 当前内容暴力分数本身就不低，保持原始分数
        result_analysis.violence_score = original_violence_score
        result_analysis.violence_type = solo_analysis.violence_type
    
    result_analysis.confidence_score = solo_analysis.confidence_score
    
    # 额外字段记录上下文分析结果，便于调试和解释
    result_analysis.context_influence = {
        "solo_violence_score": original_violence_score,
        "context_violence_score": context_analysis.violence_score,
        "full_violence_score": full_analysis.violence_score,
        "solo_sentiment": original_sentiment,
        "context_sentiment": context_analysis.sentiment,
        "influence_type": influence_type
    }
    
    print(f"最终分析结果: sentiment={result_analysis.sentiment}, violence_score={result_analysis.violence_score}")
    
    return result_analysis