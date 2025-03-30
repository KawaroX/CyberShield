// main.js
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const analyzeBtn = document.getElementById('analyze');
    const contentInput = document.getElementById('content');
    const loadingEl = document.getElementById('loading');
    const errorEl = document.getElementById('error');
    const resultEl = document.getElementById('result');
    
    // 全局变量
    let selectedContextIds = [];
    let availableTopics = [];
    let selectedContentIds = [];
    
    // 初始化Chart.js雷达图
    let microRadarChart = null;
    let topicTrendChart = null;
    
    // 页面加载时初始化
    loadTopics();
    
    // 绑定刷新话题按钮
    document.getElementById('refresh-topics').addEventListener('click', loadTopics);
    
    // 绑定添加上下文按钮
    document.getElementById('add-context').addEventListener('click', showContextDialog);
    
    // 绑定上下文对话框关闭按钮
    document.querySelector('.close').addEventListener('click', hideContextDialog);
    document.querySelector('.cancel').addEventListener('click', hideContextDialog);
    
    // 绑定搜索话题按钮
    document.getElementById('search-topics').addEventListener('click', searchTopics);
    
    // 绑定添加所选上下文按钮
    document.getElementById('add-selected-context').addEventListener('click', addSelectedContext);
    
    // 绑定话题搜索框回车事件
    document.getElementById('topic-search').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchTopics();
        }
    });
    
    // 绑定分析按钮点击事件
    analyzeBtn.addEventListener('click', analyzeWithOptions);
    
    // 加载所有话题
    function loadTopics() {
        fetch('/api/topics')
            .then(response => response.json())
            .then(data => {
                availableTopics = data.topics;
                updateTopicSelect(availableTopics);
            })
            .catch(error => console.error('加载话题失败:', error));
    }
    
    // 更新话题选择下拉框
    function updateTopicSelect(topics) {
        const select = document.getElementById('topic-select');
        
        // 保留第一个选项
        select.innerHTML = '<option value="">-- 自动分类 --</option>';
        
        // 添加话题选项
        topics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic.topic_id;
            
            // 显示话题ID和关键词
            const keywords = topic.keywords.join(', ');
            option.textContent = `${topic.topic_id} (${keywords})`;
            
            select.appendChild(option);
        });
    }
    
    // 显示上下文选择对话框
    function showContextDialog() {
        document.getElementById('context-dialog').style.display = 'block';
        document.getElementById('topic-search').focus();
    }
    
    // 隐藏上下文选择对话框
    function hideContextDialog() {
        document.getElementById('context-dialog').style.display = 'none';
    }
    
    // 搜索话题
    function searchTopics() {
        const keyword = document.getElementById('topic-search').value.trim();
        
        if (!keyword) {
            alert('请输入搜索关键词');
            return;
        }
        
        fetch(`/api/topics/search?keyword=${encodeURIComponent(keyword)}`)
            .then(response => response.json())
            .then(data => {
                displayTopicResults(data.topics);
            })
            .catch(error => console.error('搜索话题失败:', error));
    }
    
    // 显示话题搜索结果
    function displayTopicResults(topics) {
        const resultsContainer = document.querySelector('.topic-results');
        resultsContainer.innerHTML = '';
        
        if (topics.length === 0) {
            resultsContainer.innerHTML = '<div class="empty-result">未找到相关话题</div>';
            return;
        }
        
        topics.forEach(topic => {
            const topicItem = document.createElement('div');
            topicItem.className = 'topic-item';
            topicItem.dataset.topicId = topic.topic_id;
            
            const keywords = topic.keywords.join(', ');
            topicItem.innerHTML = `
                <div><strong>${topic.topic_id}</strong></div>
                <div>关键词: ${keywords}</div>
                <div>内容数量: ${topic.content_count}</div>
            `;
            
            topicItem.addEventListener('click', () => {
                // 清除其他选中
                document.querySelectorAll('.topic-item').forEach(item => {
                    item.classList.remove('selected');
                });
                
                // 选中当前项
                topicItem.classList.add('selected');
                
                // 加载话题内容
                loadTopicContents(topic.topic_id);
            });
            
            resultsContainer.appendChild(topicItem);
        });
    }
    
    // 加载话题内容
    function loadTopicContents(topicId) {
        fetch(`/api/topics/${topicId}/contents`)
            .then(response => response.json())
            .then(data => {
                displayTopicContents(data.contents);
            })
            .catch(error => console.error('加载话题内容失败:', error));
    }
    
    // 显示话题内容
    function displayTopicContents(contents) {
        const contentsContainer = document.querySelector('.topic-contents');
        contentsContainer.innerHTML = '';
        
        // 重置选中的内容
        selectedContentIds = [];
        
        if (contents.length === 0) {
            contentsContainer.innerHTML = '<div class="empty-result">该话题下没有内容</div>';
            return;
        }
        
        contents.forEach(content => {
            const contentItem = document.createElement('div');
            contentItem.className = 'content-item';
            contentItem.dataset.contentId = content.content_id;
            
            // 显示内容预览
            const preview = content.raw_content 
                ? content.raw_content.substring(0, 100) + (content.raw_content.length > 100 ? '...' : '')
                : '无内容预览';
                
            contentItem.innerHTML = `
                <div><strong>${content.content_id}</strong></div>
                <div class="content-preview">${preview}</div>
                <div>暴力分数: ${(content.violence_score * 100).toFixed(1)}%</div>
            `;
            
            contentItem.addEventListener('click', () => {
                // 切换选中状态
                contentItem.classList.toggle('selected');
                
                // 更新选中内容ID列表
                if (contentItem.classList.contains('selected')) {
                    selectedContentIds.push(content.content_id);
                } else {
                    selectedContentIds = selectedContentIds.filter(id => id !== content.content_id);
                }
            });
            
            contentsContainer.appendChild(contentItem);
        });
    }
    
    // 添加选中的上下文
    function addSelectedContext() {
        if (selectedContentIds.length === 0) {
            alert('请选择至少一个内容作为上下文');
            return;
        }
        
        // 更新选中的上下文IDs
        selectedContextIds = [...selectedContentIds];
        
        // 更新UI显示
        updateContextList();
        
        // 关闭对话框
        hideContextDialog();
    }
    
    // 更新上下文列表UI
    function updateContextList() {
        const contextList = document.getElementById('context-list');
        
        if (selectedContextIds.length === 0) {
            contextList.innerHTML = '<div class="empty-context">未选择上下文</div>';
            return;
        }
        
        contextList.innerHTML = '';
        selectedContextIds.forEach(contentId => {
            const contextItem = document.createElement('div');
            contextItem.className = 'context-item';
            
            // 显示内容ID
            contextItem.innerHTML = `
                <span class="context-text">${contentId}</span>
                <span class="remove-context" data-id="${contentId}">×</span>
            `;
            
            // 绑定移除按钮
            contextItem.querySelector('.remove-context').addEventListener('click', function() {
                const idToRemove = this.dataset.id;
                selectedContextIds = selectedContextIds.filter(id => id !== idToRemove);
                updateContextList();
            });
            
            contextList.appendChild(contextItem);
        });
    }
    
    // 分析内容（带选项）
    function analyzeWithOptions() {
        const content = contentInput.value.trim();
        if (!content) {
            alert('请输入要分析的内容');
            return;
        }
        
        // 获取选项
        const targetTopicId = document.getElementById('topic-select').value;
        const useContext = selectedContextIds.length > 0;
        
        // 显示加载状态
        loadingEl.style.display = 'block';
        errorEl.style.display = 'none';
        resultEl.style.display = 'none';
        
        // 根据选项决定使用哪个API
        let apiUrl = '/api/analyze';
        let requestData = {
            content: content,
            content_type: 'text'
        };
        
        if (targetTopicId) {
            // 使用指定话题
            requestData.topic_id = targetTopicId;
            console.log("使用指定话题:", targetTopicId);
        } else if (useContext) {
            // 使用上下文
            apiUrl = '/api/analyze_with_context';
            requestData.context_ids = selectedContextIds;
            console.log("使用上下文:", selectedContextIds);
        }
        
        // 调用API
        console.log("发送请求:", apiUrl, requestData);
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('分析失败，服务器返回错误: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // 隐藏加载状态，显示结果
            loadingEl.style.display = 'none';
            resultEl.style.display = 'block';
            
            // 更新UI
            updateAnalysisResults(data);
        })
        .catch(error => {
            // 隐藏加载状态，显示错误
            loadingEl.style.display = 'none';
            errorEl.textContent = '分析失败: ' + error.message;
            errorEl.style.display = 'block';
            console.error('Error:', error);
        });
    }
    
    // 更新分析结果UI
    function updateAnalysisResults(data) {
        // 微观分析
        updateMicroAnalysis(data.micro_analysis);
        
        // 微观操作
        updateMicroAction(data.micro_action);
        
        // 宏观分析
        updateMacroAnalysis(data.macro_analysis);
        
        // 宏观干预
        updateMacroInterventions(data.macro_interventions);
        
        // 更新雷达图
        updateCharts(data);

        if (data.context_influence) {
            const influenceSection = document.createElement('div');
            influenceSection.className = 'context-influence-section';
            
            // 计算上下文影响程度
            const soloScore = data.context_influence.solo_violence_score;
            const finalScore = data.micro_analysis.violence_score;
            const difference = finalScore - soloScore;
            
            let influenceText = '';
            if (Math.abs(difference) < 0.1) {
                influenceText = '上下文对分析结果没有显著影响';
            } else if (difference > 0) {
                influenceText = `上下文使暴力分数增加了 ${(difference * 100).toFixed(1)}%`;
            } else {
                influenceText = `上下文使暴力分数减少了 ${(Math.abs(difference) * 100).toFixed(1)}%`;
            }
            
            influenceSection.innerHTML = `
                <div class="influence-header">上下文影响分析</div>
                <div class="influence-content">
                    <div>${influenceText}</div>
                    <div class="influence-details">
                        <div>单独内容暴力分数: ${(soloScore * 100).toFixed(1)}%</div>
                        <div>考虑上下文后暴力分数: ${(finalScore * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
            
            // 添加到结果区域
            const microAnalysisCard = document.querySelector('.result-card:first-child');
            microAnalysisCard.appendChild(influenceSection);
        }
    }
    
    // 更新微观分析
    function updateMicroAnalysis(microAnalysis) {
        // 暴力分数
        const violenceScore = (microAnalysis.violence_score * 100).toFixed(1);
        document.getElementById('violence-score').textContent = violenceScore + '%';
        document.getElementById('violence-fill').style.width = violenceScore + '%';
        
        // 设置暴力分数条的颜色
        const violenceMeter = document.getElementById('violence-meter');
        violenceMeter.className = 'meter-container';
        if (microAnalysis.violence_score > 0.7) {
            violenceMeter.classList.add('high-risk');
        } else if (microAnalysis.violence_score > 0.4) {
            violenceMeter.classList.add('medium-risk');
        } else {
            violenceMeter.classList.add('low-risk');
        }
        
        // 设置暴力类型、置信度和是否负面
        document.getElementById('violence-type').textContent = microAnalysis.violence_type || '无';
        document.getElementById('confidence-score').textContent = (microAnalysis.confidence_score * 100).toFixed(1) + '%';
        document.getElementById('is-negative').textContent = microAnalysis.is_negative ? '是' : '否';
        
        // 设置情感分析结果
        const sentimentEmoji = document.getElementById('sentiment-emoji');
        const sentimentMarker = document.getElementById('sentiment-marker');
        
        if (microAnalysis.sentiment === 0) {
            sentimentEmoji.textContent = '😟';
            sentimentMarker.style.left = '20%';
        } else if (microAnalysis.sentiment === 2) {
            sentimentEmoji.textContent = '😊';
            sentimentMarker.style.left = '80%';
        } else {
            sentimentEmoji.textContent = '😐';
            sentimentMarker.style.left = '50%';
        }
        
        // 设置关键词
        const microKeywords = document.getElementById('micro-keywords');
        microKeywords.innerHTML = '';
        if (microAnalysis.keywords && microAnalysis.keywords.length > 0) {
            microAnalysis.keywords.forEach(keyword => {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = keyword;
                microKeywords.appendChild(tag);
            });
        } else {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.textContent = '无';
            microKeywords.appendChild(tag);
        }
    }
    
    // 更新微观操作
    function updateMicroAction(microAction) {
        document.getElementById('action-type').textContent = getActionTypeText(microAction.action_type);
        document.getElementById('severity').textContent = getSeverityText(microAction.severity);
        document.getElementById('action-message').textContent = microAction.message || '无';
    }
    
    // 更新宏观分析
    function updateMacroAnalysis(macroAnalysis) {
        const riskScore = (macroAnalysis.violence_risk_score * 100).toFixed(1);
        document.getElementById('risk-score').textContent = riskScore + '%';
        document.getElementById('risk-fill').style.width = riskScore + '%';
        
        const riskMeter = document.getElementById('risk-meter');
        riskMeter.className = 'meter-container';
        if (macroAnalysis.violence_risk_score > 0.6) {
            riskMeter.classList.add('high-risk');
        } else if (macroAnalysis.violence_risk_score > 0.3) {
            riskMeter.classList.add('medium-risk');
        } else {
            riskMeter.classList.add('low-risk');
        }
        
        document.getElementById('topic-id').textContent = macroAnalysis.topic_id;
        document.getElementById('content-count').textContent = macroAnalysis.content_count;
        document.getElementById('intervention-status').textContent = getInterventionStatusText(macroAnalysis.intervention_status);
        
        // 设置宏观关键词
        const macroKeywords = document.getElementById('macro-keywords');
        macroKeywords.innerHTML = '';
        if (macroAnalysis.keywords && macroAnalysis.keywords.length > 0) {
            macroAnalysis.keywords.forEach(keyword => {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = keyword;
                macroKeywords.appendChild(tag);
            });
        } else {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.textContent = '无';
            macroKeywords.appendChild(tag);
        }
    }
    
    // 更新宏观干预
    function updateMacroInterventions(interventions) {
        const interventionList = document.getElementById('intervention-list');
        interventionList.innerHTML = '';
        
        if (interventions && interventions.length > 0) {
            interventions.forEach(intervention => {
                const li = document.createElement('li');
                li.className = 'intervention-item';
                
                // 确保priority是字符串并转为小写
                const priorityValue = String(intervention.priority || '').toLowerCase();
                const priorityClass = `priority-${priorityValue}`;
                
                li.innerHTML = `
                    <span class="intervention-priority ${priorityClass}">${getInterventionPriorityText(intervention.priority)}</span>
                    <span>${intervention.description}</span>
                `;
                
                interventionList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.className = 'intervention-item';
            li.textContent = '无干预建议';
            interventionList.appendChild(li);
        }
    }
    
    // 更新图表
    function updateCharts(data) {
        // 更新微观雷达图
        updateMicroRadarChart(data);
        
        // 更新话题趋势图
        updateTopicTrendChart(data);
    }
    
    // 更新微观雷达图
    function updateMicroRadarChart(data) {
        const ctx = document.getElementById('micro-radar-chart').getContext('2d');
        
        // 销毁现有图表（如果存在）
        if (microRadarChart) {
            microRadarChart.destroy();
        }
        
        // 准备数据
        const radarData = {
            labels: ['暴力分数', '负面情绪', '严重程度', '干预必要性', '影响范围'],
            datasets: [{
                label: '内容分析',
                data: [
                    data.micro_analysis.violence_score * 100,
                    data.micro_analysis.negative_prob ? data.micro_analysis.negative_prob * 100 : 50,
                    getSeverityValue(data.micro_action.severity),
                    getInterventionValue(data.macro_analysis.intervention_status),
                    data.macro_analysis.violence_risk_score * 100
                ],
                backgroundColor: 'rgba(245, 245, 245, 0.2)',
                borderColor: 'rgba(245, 245, 245, 0.8)',
                pointBackgroundColor: 'rgba(245, 245, 245, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(245, 245, 245, 1)'
            }]
        };
        
        // 创建雷达图
        microRadarChart = new Chart(ctx, {
            type: 'radar',
            data: radarData,
            options: {
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                family: "'Work Sans', sans-serif",
                                size: 12
                            }
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.5)',
                            backdropColor: 'transparent',
                            font: {
                                size: 10
                            }
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // 更新话题趋势图
    function updateTopicTrendChart(data) {
        const ctx = document.getElementById('topic-trend-chart').getContext('2d');
        
        // 销毁现有图表（如果存在）
        if (topicTrendChart) {
            topicTrendChart.destroy();
        }
        
        // 这里我们使用一个简单的雷达图来表示话题的不同维度
        const radarData = {
            labels: ['负面比例', '用户参与度', '增长速率', '暴力风险', '干预难度'],
            datasets: [{
                label: '话题分析',
                data: [
                    data.macro_analysis.negativity_ratio * 100,
                    Math.min(data.macro_analysis.content_count * 10, 100), // 简化的用户参与度计算
                    Math.min(data.macro_analysis.growth_rate * 100 || 20, 100), // 简化的增长速率
                    data.macro_analysis.violence_risk_score * 100,
                    getInterventionDifficulty(data.macro_analysis.intervention_status)
                ],
                backgroundColor: 'rgba(255, 152, 0, 0.2)',
                borderColor: 'rgba(255, 152, 0, 0.8)',
                pointBackgroundColor: 'rgba(255, 152, 0, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(255, 152, 0, 1)'
            }]
        };
        
        // 创建雷达图
        topicTrendChart = new Chart(ctx, {
            type: 'radar',
            data: radarData,
            options: {
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                family: "'Work Sans', sans-serif",
                                size: 12
                            }
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.5)',
                            backdropColor: 'transparent',
                            font: {
                                size: 10
                            }
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // 辅助函数：获取操作类型文本
    function getActionTypeText(actionType) {
        const actionTypeMap = {
            'remove': '移除内容',
            'restrict': '限制访问',
            'warning': '发出警告',
            'flag': '标记审核',
            'none': '无操作'
        };
        
        return actionTypeMap[actionType] || actionType;
    }
    
    // 辅助函数：获取严重程度文本
    function getSeverityText(severity) {
        const severityMap = {
            'critical': '严重',
            'high': '高',
            'medium': '中',
            'low': '低',
            'none': '无'
        };
        
        return severityMap[severity] || severity;
    }
    
    // 辅助函数：获取干预状态文本
    function getInterventionStatusText(status) {
        const statusMap = {
            'Monitoring': '监控中',
            'EarlyWarning': '预警中',
            'ActiveIntervention': '干预中'
        };
        
        return statusMap[status] || status;
    }
    
    // 辅助函数：获取干预优先级文本
    function getInterventionPriorityText(priority) {
        const priorityMap = {
            'high': '高',
            'medium': '中',
            'low': '低'
        };
        
        return priorityMap[priority] || priority;
    }
    
    // 辅助函数：获取严重程度数值（用于图表）
    function getSeverityValue(severity) {
        const severityMap = {
            'critical': 90,
            'high': 70,
            'medium': 50,
            'low': 30,
            'none': 10
        };
        
        return severityMap[severity] || 0;
    }
    
    // 辅助函数：获取干预状态数值（用于图表）
    function getInterventionValue(status) {
        const statusMap = {
            'ActiveIntervention': 90,
            'EarlyWarning': 60,
            'Monitoring': 30
        };
        
        return statusMap[status] || 0;
    }
    
    // 辅助函数：获取干预难度（用于图表）
    function getInterventionDifficulty(status) {
        const difficultyMap = {
            'ActiveIntervention': 80,
            'EarlyWarning': 50,
            'Monitoring': 20
        };
        
        return difficultyMap[status] || 0;
    }
});