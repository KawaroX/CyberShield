// CyberShield 修复后的主JavaScript文件
document.addEventListener('DOMContentLoaded', function() {
    // DOM元素
    const analyzeBtn = document.getElementById('analyze-btn');
    const contentInput = document.getElementById('content-input');
    const loadingEl = document.getElementById('loading');
    const resultsEl = document.getElementById('results');
    const contextModal = document.getElementById('context-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const cancelContextBtn = document.getElementById('cancel-context');
    const addContextBtn = document.getElementById('add-context');
    const searchTopicsBtn = document.getElementById('search-topics');
    const addSelectedContextBtn = document.getElementById('add-selected-context');
    const refreshTopicsBtn = document.getElementById('refresh-topics');
    const topicSearchInput = document.getElementById('topic-search');
    
    // 全局变量
    let selectedContextIds = [];
    let availableTopics = [];
    let selectedContentIds = [];
    
    // 图表实例
    let microRadarChart = null;
    let topicTrendChart = null;
    
    // 初始化
    init();
    
    // 初始化函数
    function init() {
        // 加载话题
        loadTopics();
        
        // 绑定事件
        bindEvents();
        
        // 显示功能开发状态通知
        showNotification('系统提示', '部分功能（监控面板、话题分析、设置、帮助）正在开发中', 'info');
        
        // 添加主题切换按钮
        addThemeToggle();
    }
    
    // 添加主题切换按钮
    function addThemeToggle() {
        const navbar = document.querySelector('.navbar');
        if (navbar) {
            const themeToggle = document.createElement('div');
            themeToggle.className = 'theme-switch';
            themeToggle.innerHTML = `
                <div class="theme-switch-track"></div>
                <div class="theme-switch-thumb"></div>
            `;
            
            // 检查当前主题
            const currentTheme = localStorage.getItem('theme');
            if (currentTheme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
                themeToggle.classList.add('light');
            }
            
            // 添加切换事件
            themeToggle.addEventListener('click', function() {
                if (document.documentElement.getAttribute('data-theme') === 'light') {
                    document.documentElement.setAttribute('data-theme', 'dark');
                    localStorage.setItem('theme', 'dark');
                    themeToggle.classList.remove('light');
                    showNotification('主题已切换', '已切换到暗色模式', 'success');
                } else {
                    document.documentElement.setAttribute('data-theme', 'light');
                    localStorage.setItem('theme', 'light');
                    themeToggle.classList.add('light');
                    showNotification('主题已切换', '已切换到亮色模式', 'success');
                }
            });
            
            navbar.appendChild(themeToggle);
        }
    }
    
    // 绑定事件处理函数
    function bindEvents() {
        // 分析按钮
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', analyzeContent);
        }
        
        // 上下文相关
        if (addContextBtn) {
            addContextBtn.addEventListener('click', openContextModal);
        }
        
        if (closeModalBtn) {
            closeModalBtn.addEventListener('click', closeContextModal);
        }
        
        if (cancelContextBtn) {
            cancelContextBtn.addEventListener('click', closeContextModal);
        }
        
        if (searchTopicsBtn) {
            searchTopicsBtn.addEventListener('click', searchTopics);
        }
        
        if (addSelectedContextBtn) {
            addSelectedContextBtn.addEventListener('click', addSelectedContext);
        }
        
        // 话题搜索框回车触发搜索
        if (topicSearchInput) {
            topicSearchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    searchTopics();
                }
            });
        }
        
        // 刷新话题按钮
        if (refreshTopicsBtn) {
            refreshTopicsBtn.addEventListener('click', loadTopics);
        }
        
        // 顶部导航链接处理
        document.querySelectorAll('.nav-link:not(.active)').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                showNotification('开发中', '该功能正在开发中，敬请期待', 'info');
            });
        });
    }
    
    // 加载话题
    function loadTopics() {
        fetch('/api/topics')
            .then(response => response.json())
            .then(data => {
                availableTopics = data.topics;
                updateTopicSelect(availableTopics);
            })
            .catch(error => {
                console.error('加载话题失败:', error);
                showNotification('错误', '加载话题列表失败，请稍后重试', 'error');
            });
    }
    
    // 更新话题选择下拉框
    function updateTopicSelect(topics) {
        const select = document.getElementById('topic-select');
        if (!select) return;
        
        // 保留第一个选项
        select.innerHTML = '<option value="">-- 自动分类 --</option>';
        
        // 添加话题选项
        topics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic.topic_id;
            
            // 显示话题ID和关键词
            const keywords = topic.keywords ? topic.keywords.join(', ') : '无关键词';
            option.textContent = `${topic.topic_id} (${keywords})`;
            
            select.appendChild(option);
        });
    }
    
    // 打开上下文选择对话框
    function openContextModal() {
        if (contextModal) {
            contextModal.style.display = 'flex';
            if (topicSearchInput) {
                topicSearchInput.focus();
            }
        }
    }
    
    // 关闭上下文选择对话框
    function closeContextModal() {
        if (contextModal) {
            contextModal.style.display = 'none';
        }
    }
    
    // 搜索话题
    function searchTopics() {
        if (!topicSearchInput) return;
        
        const keyword = topicSearchInput.value.trim();
        
        if (!keyword) {
            showNotification('提示', '请输入搜索关键词', 'warning');
            return;
        }
        
        const topicResultsEl = document.getElementById('topic-results');
        if (topicResultsEl) {
            topicResultsEl.innerHTML = '<div class="empty-message">搜索中...</div>';
        }
        
        fetch(`/api/topics/search?keyword=${encodeURIComponent(keyword)}`)
            .then(response => response.json())
            .then(data => {
                displayTopicResults(data.topics);
            })
            .catch(error => {
                console.error('搜索话题失败:', error);
                if (topicResultsEl) {
                    topicResultsEl.innerHTML = '<div class="empty-message">搜索失败，请重试</div>';
                }
            });
    }
    
    // 显示话题搜索结果
    function displayTopicResults(topics) {
        const resultsContainer = document.getElementById('topic-results');
        if (!resultsContainer) return;
        
        resultsContainer.innerHTML = '';
        
        if (!topics || topics.length === 0) {
            resultsContainer.innerHTML = '<div class="empty-message">未找到相关话题</div>';
            return;
        }
        
        topics.forEach(topic => {
            const topicItem = document.createElement('div');
            topicItem.className = 'result-item';
            topicItem.dataset.topicId = topic.topic_id;
            
            const keywords = topic.keywords ? topic.keywords.join(', ') : '无关键词';
            topicItem.innerHTML = `
                <div><strong>${topic.topic_id}</strong></div>
                <div>关键词: ${keywords}</div>
                <div>内容数量: ${topic.content_count || 0}</div>
            `;
            
            topicItem.addEventListener('click', () => {
                // 清除其他选中状态
                document.querySelectorAll('.result-item').forEach(item => {
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
        const contentsContainer = document.getElementById('topic-contents');
        if (!contentsContainer) return;
        
        contentsContainer.innerHTML = '<div class="empty-message">加载中...</div>';
        
        fetch(`/api/topics/${topicId}/contents`)
            .then(response => response.json())
            .then(data => {
                displayTopicContents(data.contents);
            })
            .catch(error => {
                console.error('加载话题内容失败:', error);
                contentsContainer.innerHTML = '<div class="empty-message">加载失败，请重试</div>';
            });
    }
    
    // 显示话题内容
    function displayTopicContents(contents) {
        const contentsContainer = document.getElementById('topic-contents');
        if (!contentsContainer) return;
        
        contentsContainer.innerHTML = '';
        
        // 重置选中的内容
        selectedContentIds = [];
        
        if (!contents || contents.length === 0) {
            contentsContainer.innerHTML = '<div class="empty-message">该话题下没有内容</div>';
            return;
        }
        
        contents.forEach(content => {
            const contentItem = document.createElement('div');
            contentItem.className = 'result-item';
            contentItem.dataset.contentId = content.content_id;
            
            // 显示内容预览
            const preview = content.raw_content 
                ? content.raw_content.substring(0, 80) + (content.raw_content.length > 80 ? '...' : '')
                : '无内容预览';
                
            const violenceScore = content.violence_score !== undefined 
                ? (content.violence_score * 100).toFixed(1) + '%'
                : '未知';
                
            contentItem.innerHTML = `
                <div><strong>${content.content_id}</strong></div>
                <div class="content-preview">${preview}</div>
                <div>暴力分数: ${violenceScore}</div>
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
            showNotification('提示', '请选择至少一个内容作为上下文', 'warning');
            return;
        }
        
        // 更新选中的上下文IDs
        selectedContextIds = [...selectedContentIds];
        
        // 更新UI显示
        updateContextList();
        
        // 关闭对话框
        closeContextModal();
        
        showNotification('成功', `已添加 ${selectedContentIds.length} 个上下文内容`, 'success');
    }
    
    // 更新上下文列表UI
    function updateContextList() {
        const contextList = document.getElementById('context-list');
        if (!contextList) return;
        
        if (selectedContextIds.length === 0) {
            contextList.innerHTML = '<div class="empty-message">未选择上下文内容</div>';
            return;
        }
        
        contextList.innerHTML = '';
        selectedContextIds.forEach(contentId => {
            const contextItem = document.createElement('div');
            contextItem.className = 'context-item';
            
            // 显示内容ID
            contextItem.innerHTML = `
                <span class="context-text">${contentId}</span>
                <button class="remove-btn" data-id="${contentId}">×</button>
            `;
            
            // 绑定移除按钮
            contextItem.querySelector('.remove-btn').addEventListener('click', function() {
                const idToRemove = this.dataset.id;
                selectedContextIds = selectedContextIds.filter(id => id !== idToRemove);
                updateContextList();
            });
            
            contextList.appendChild(contextItem);
        });
    }
    
    // 分析内容
    function analyzeContent() {
        if (!contentInput) return;
        
        const content = contentInput.value.trim();
        if (!content) {
            showNotification('提示', '请输入要分析的内容', 'warning');
            return;
        }
        
        // 获取选项
        const targetTopicId = document.getElementById('topic-select')?.value;
        const useContext = selectedContextIds.length > 0;
        
        // 显示加载状态
        if (loadingEl) loadingEl.style.display = 'flex';
        if (resultsEl) resultsEl.style.display = 'none';
        
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
            requestData.context = selectedContextIds.map(id => ({
                content_id: id,
                timestamp: new Date().toISOString()
            }));
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
            if (loadingEl) loadingEl.style.display = 'none';
            if (resultsEl) resultsEl.style.display = 'grid';
            
            // 更新UI
            updateAnalysisResults(data);
            
            // 滚动到结果区域
            if (resultsEl) resultsEl.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // 隐藏加载状态，显示错误
            if (loadingEl) loadingEl.style.display = 'none';
            console.error('分析失败:', error);
            showNotification('错误', '分析失败: ' + error.message, 'error');
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
        
        // 更新图表
        updateCharts(data);
        
        // 更新上下文影响 (如果有)
        if (data.context_influence) {
            updateContextInfluence(data.context_influence, data.micro_analysis.violence_score);
        } else {
            const contextInfluenceEl = document.getElementById('context-influence');
            if (contextInfluenceEl) contextInfluenceEl.style.display = 'none';
        }
    }
    
    // 更新微观分析
    function updateMicroAnalysis(microAnalysis) {
        // 暴力分数
        const violenceScore = (microAnalysis.violence_score * 100).toFixed(1);
        const violenceScoreEl = document.getElementById('violence-score');
        const violenceFillEl = document.getElementById('violence-fill');
        
        if (violenceScoreEl) violenceScoreEl.textContent = violenceScore + '%';
        if (violenceFillEl) violenceFillEl.style.width = violenceScore + '%';
        
        // 设置暴力分数条的颜色
        if (violenceFillEl) {
            violenceFillEl.className = 'progress-fill';
            if (microAnalysis.violence_score > 0.7) {
                violenceFillEl.classList.add('progress-high');
            } else if (microAnalysis.violence_score > 0.4) {
                violenceFillEl.classList.add('progress-medium');
            } else {
                violenceFillEl.classList.add('progress-low');
            }
        }
        
        // 设置暴力类型、置信度和是否负面
        const violenceTypeEl = document.getElementById('violence-type');
        const confidenceScoreEl = document.getElementById('confidence-score');
        const isNegativeEl = document.getElementById('is-negative');
        
        if (violenceTypeEl) violenceTypeEl.textContent = microAnalysis.violence_type || '无';
        if (confidenceScoreEl) confidenceScoreEl.textContent = (microAnalysis.confidence_score * 100).toFixed(1) + '%';
        if (isNegativeEl) isNegativeEl.textContent = microAnalysis.is_negative ? '是' : '否';
        
        // 设置情感分析结果
        const sentimentEmoji = document.getElementById('sentiment-emoji');
        const sentimentMarker = document.getElementById('sentiment-marker');
        
        if (sentimentEmoji && sentimentMarker) {
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
        }
        
        // 设置关键词
        const microKeywords = document.getElementById('micro-keywords');
        if (microKeywords) {
            microKeywords.innerHTML = '';
            if (microAnalysis.keywords && microAnalysis.keywords.length > 0) {
                microAnalysis.keywords.forEach(keyword => {
                    if (keyword) {
                        const tag = document.createElement('span');
                        tag.className = 'tag';
                        tag.textContent = keyword;
                        microKeywords.appendChild(tag);
                    }
                });
            } else {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = '无关键词';
                microKeywords.appendChild(tag);
            }
        }
    }
    
    // 更新微观操作
    function updateMicroAction(microAction) {
        const actionTypeEl = document.getElementById('action-type');
        const severityEl = document.getElementById('severity');
        const actionMessageEl = document.getElementById('action-message');
        
        if (actionTypeEl) actionTypeEl.textContent = getActionTypeText(microAction.action_type);
        if (severityEl) severityEl.textContent = getSeverityText(microAction.severity);
        if (actionMessageEl) actionMessageEl.textContent = microAction.message || '无';
    }
    
    // 更新宏观分析
    function updateMacroAnalysis(macroAnalysis) {
        const riskScoreEl = document.getElementById('risk-score');
        const riskFillEl = document.getElementById('risk-fill');
        
        if (riskScoreEl && riskFillEl) {
            const riskScore = (macroAnalysis.violence_risk_score * 100).toFixed(1);
            riskScoreEl.textContent = riskScore + '%';
            riskFillEl.style.width = riskScore + '%';
            
            riskFillEl.className = 'progress-fill';
            if (macroAnalysis.violence_risk_score > 0.6) {
                riskFillEl.classList.add('progress-high');
            } else if (macroAnalysis.violence_risk_score > 0.3) {
                riskFillEl.classList.add('progress-medium');
            } else {
                riskFillEl.classList.add('progress-low');
            }
        }
        
        const topicIdEl = document.getElementById('topic-id');
        const contentCountEl = document.getElementById('content-count');
        const interventionStatusEl = document.getElementById('intervention-status');
        
        if (topicIdEl) topicIdEl.textContent = macroAnalysis.topic_id;
        if (contentCountEl) contentCountEl.textContent = macroAnalysis.content_count;
        if (interventionStatusEl) interventionStatusEl.textContent = getInterventionStatusText(macroAnalysis.intervention_status);
        
        // 设置宏观关键词
        const macroKeywords = document.getElementById('macro-keywords');
        if (macroKeywords) {
            macroKeywords.innerHTML = '';
            if (macroAnalysis.keywords && macroAnalysis.keywords.length > 0) {
                macroAnalysis.keywords.forEach(keyword => {
                    if (keyword) {
                        const tag = document.createElement('span');
                        tag.className = 'tag';
                        tag.textContent = keyword;
                        macroKeywords.appendChild(tag);
                    }
                });
            } else {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = '无关键词';
                macroKeywords.appendChild(tag);
            }
        }
    }
    
    // 更新宏观干预
    function updateMacroInterventions(interventions) {
        const interventionList = document.getElementById('intervention-list');
        if (!interventionList) return;
        
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
                    <span class="intervention-text">${intervention.description}</span>
                `;
                
                interventionList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.className = 'intervention-item';
            li.innerHTML = '<span class="intervention-text">无干预建议</span>';
            interventionList.appendChild(li);
        }
    }
    
    // 更新上下文影响分析
    function updateContextInfluence(contextInfluence, finalScore) {
        const contextInfluenceEl = document.getElementById('context-influence');
        const influenceMessageEl = document.getElementById('influence-message');
        const soloScoreEl = document.getElementById('solo-score');
        const finalScoreEl = document.getElementById('final-score');
        
        if (!contextInfluenceEl || !influenceMessageEl || !soloScoreEl || !finalScoreEl) return;
        
        // 显示上下文影响区域
        contextInfluenceEl.style.display = 'block';
        
        // 提取数值
        const soloScore = contextInfluence.solo_violence_score;
        const difference = finalScore - soloScore;
        
        // 设置影响信息
        let influenceText = '';
        let influenceClass = '';
        
        if (Math.abs(difference) < 0.1) {
            influenceText = '上下文对分析结果没有显著影响';
            influenceClass = 'neutral';
        } else if (difference > 0) {
            influenceText = `上下文使暴力风险增加，该内容在当前上下文中可能更具危险性`;
            influenceClass = 'increased';
        } else {
            influenceText = `上下文降低了暴力风险，该内容在当前上下文中危险性较低`;
            influenceClass = 'decreased';
        }
        
        influenceMessageEl.textContent = influenceText;
        influenceMessageEl.className = `influence-message ${influenceClass}`;
        
        // 设置详细数据
        soloScoreEl.textContent = `单独分析暴力分数: ${(soloScore * 100).toFixed(1)}%`;
        finalScoreEl.textContent = `考虑上下文后暴力分数: ${(finalScore * 100).toFixed(1)}%`;
        
        if (difference > 0) {
            soloScoreEl.innerHTML = `单独分析暴力分数: ${(soloScore * 100).toFixed(1)}% <i class="fas fa-arrow-up influence-stat-icon"></i>`;
            soloScoreEl.className = 'influence-stat increased';
        } else if (difference < 0) {
            soloScoreEl.innerHTML = `单独分析暴力分数: ${(soloScore * 100).toFixed(1)}% <i class="fas fa-arrow-down influence-stat-icon"></i>`;
            soloScoreEl.className = 'influence-stat decreased';
        } else {
            soloScoreEl.className = 'influence-stat neutral';
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
        const ctx = document.getElementById('micro-radar-chart')?.getContext('2d');
        if (!ctx) return;
        
        // 销毁现有图表（如果存在）
        if (microRadarChart) {
            microRadarChart.destroy();
        }
        
        // 准备数据
        const radarData = {
            labels: ['暴力分数', '负面情绪', '干预必要性', '影响范围', '严重程度'],
            datasets: [{
                label: '内容分析',
                data: [
                    data.micro_analysis.violence_score * 100,
                    data.micro_analysis.negative_prob ? data.micro_analysis.negative_prob * 100 : 50,
                    getInterventionValue(data.macro_analysis.intervention_status),
                    data.macro_analysis.violence_risk_score * 100,
                    getSeverityValue(data.micro_action.severity)
                ],
                backgroundColor: 'rgba(56, 151, 255, 0.2)',
                borderColor: 'rgba(56, 151, 255, 0.8)',
                pointBackgroundColor: '#3897ff',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#3897ff'
            }]
        };
        
        // 创建雷达图并修复显示问题
        microRadarChart = new Chart(ctx, {
            type: 'radar',
            data: radarData,
            options: {
                responsive: true,
                maintainAspectRatio: true, // 保持图表的宽高比
                elements: {
                    line: {
                        borderWidth: 2
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: document.documentElement.getAttribute('data-theme') === 'light' 
                                ? 'rgba(0, 0, 0, 0.1)' 
                                : 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: document.documentElement.getAttribute('data-theme') === 'light' 
                                ? 'rgba(0, 0, 0, 0.1)' 
                                : 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: document.documentElement.getAttribute('data-theme') === 'light'
                                ? 'rgba(45, 55, 72, 0.7)'
                                : 'rgba(255, 255, 255, 0.7)',
                            font: {
                                family: "'Noto Sans SC', sans-serif",
                                size: 12
                            }
                        },
                        ticks: {
                            color: document.documentElement.getAttribute('data-theme') === 'light'
                                ? 'rgba(45, 55, 72, 0.5)'
                                : 'rgba(255, 255, 255, 0.5)',
                            backdropColor: 'transparent',
                            font: {
                                size: 10
                            },
                            display: false
                        },
                        suggestedMin: 0,
                        suggestedMax: 100,
                        beginAtZero: true // 确保从0开始
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? 'rgba(255, 255, 255, 0.9)'
                            : 'rgba(0, 0, 0, 0.7)',
                        titleColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? '#2d3748'
                            : '#ffffff',
                        bodyColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? '#4a5568'
                            : '#a8b2c1',
                        borderColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? '#cbd5e0'
                            : '#3b4252',
                        borderWidth: 1,
                        displayColors: false,
                        padding: 10,
                        titleFont: {
                            family: "'Noto Sans SC', sans-serif",
                            size: 14,
                            weight: 'bold'
                        },
                        bodyFont: {
                            family: "'Noto Sans SC', sans-serif",
                            size: 13
                        },
                        callbacks: {
                            title: function(tooltipItems) {
                                return radarData.labels[tooltipItems[0].dataIndex];
                            },
                            label: function(context) {
                                return context.raw.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // 更新话题趋势图
    function updateTopicTrendChart(data) {
        const ctx = document.getElementById('topic-trend-chart')?.getContext('2d');
        if (!ctx) return;
        
        // 销毁现有图表（如果存在）
        if (topicTrendChart) {
            topicTrendChart.destroy();
        }
        
        // 使用雷达图表示话题的不同维度
        const radarData = {
            labels: ['负面比例', '用户参与度', '传播速率', '暴力风险', '干预难度'],
            datasets: [{
                label: '话题分析',
                data: [
                    data.macro_analysis.negativity_ratio * 100,
                    Math.min(data.macro_analysis.content_count * 10, 100), // 简化的用户参与度计算
                    Math.min(data.macro_analysis.growth_rate * 100 || 20, 100), // 简化的增长速率
                    data.macro_analysis.violence_risk_score * 100,
                    getInterventionDifficulty(data.macro_analysis.intervention_status)
                ],
                backgroundColor: 'rgba(157, 122, 255, 0.2)',
                borderColor: 'rgba(157, 122, 255, 0.8)',
                pointBackgroundColor: '#9d7aff',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#9d7aff'
            }]
        };
        
        // 创建雷达图并修复显示问题
        topicTrendChart = new Chart(ctx, {
            type: 'radar',
            data: radarData,
            options: {
                responsive: true,
                maintainAspectRatio: true, // 保持图表的宽高比
                elements: {
                    line: {
                        borderWidth: 2
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: document.documentElement.getAttribute('data-theme') === 'light' 
                                ? 'rgba(0, 0, 0, 0.1)' 
                                : 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: document.documentElement.getAttribute('data-theme') === 'light' 
                                ? 'rgba(0, 0, 0, 0.1)' 
                                : 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: document.documentElement.getAttribute('data-theme') === 'light'
                                ? 'rgba(45, 55, 72, 0.7)'
                                : 'rgba(255, 255, 255, 0.7)',
                            font: {
                                family: "'Noto Sans SC', sans-serif",
                                size: 12
                            }
                        },
                        ticks: {
                            color: document.documentElement.getAttribute('data-theme') === 'light'
                                ? 'rgba(45, 55, 72, 0.5)'
                                : 'rgba(255, 255, 255, 0.5)',
                            backdropColor: 'transparent',
                            font: {
                                size: 10
                            },
                            display: false
                        },
                        suggestedMin: 0,
                        suggestedMax: 100,
                        beginAtZero: true // 确保从0开始
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? 'rgba(255, 255, 255, 0.9)'
                            : 'rgba(0, 0, 0, 0.7)',
                        titleColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? '#2d3748'
                            : '#ffffff',
                        bodyColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? '#4a5568'
                            : '#a8b2c1',
                        borderColor: document.documentElement.getAttribute('data-theme') === 'light'
                            ? '#cbd5e0'
                            : '#3b4252',
                        borderWidth: 1,
                        displayColors: false,
                        padding: 10,
                        titleFont: {
                            family: "'Noto Sans SC', sans-serif",
                            size: 14,
                            weight: 'bold'
                        },
                        bodyFont: {
                            family: "'Noto Sans SC', sans-serif",
                            size: 13
                        },
                        callbacks: {
                            title: function(tooltipItems) {
                                return radarData.labels[tooltipItems[0].dataIndex];
                            },
                            label: function(context) {
                                return context.raw.toFixed(1) + '%';
                            }
                        }
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
    
    // 显示通知
    function showNotification(title, message, type = 'info') {
        // 检查是否已存在通知，如果存在则移除
        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }
        
        // 创建新通知
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        // 根据类型设置图标
        let icon = 'info-circle';
        if (type === 'success') icon = 'check-circle';
        if (type === 'warning') icon = 'exclamation-triangle';
        if (type === 'error') icon = 'times-circle';
        
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas fa-${icon}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
        `;
        
        // 添加到文档
        document.body.appendChild(notification);
        
        // 4秒后自动移除
        setTimeout(() => {
            notification.remove();
        }, 4000);
    }
});