document.addEventListener('DOMContentLoaded', function() {
    // 主题切换功能
    setupThemeToggle();
    
    // 数据可视化增强
    enhanceVisualizations();
    
    // 监控面板实时更新
    setupMonitoringPanel();
    
    // 首页动画效果
    setupHomeAnimations();
});

// 设置主题切换
function setupThemeToggle() {
    // 检查用户是否已经有偏好设置
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        // 如果用户系统偏好浅色模式
        document.documentElement.setAttribute('data-theme', 'light');
    }
    
    // 添加主题切换按钮
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        const themeToggle = document.createElement('div');
        themeToggle.className = 'theme-switch';
        themeToggle.innerHTML = `
            <div class="theme-switch-track"></div>
            <div class="theme-switch-thumb"></div>
        `;
        
        // 设置当前状态
        if (document.documentElement.getAttribute('data-theme') === 'light') {
            themeToggle.classList.add('light');
        }
        
        // 添加切换事件
        themeToggle.addEventListener('click', function() {
            if (document.documentElement.getAttribute('data-theme') === 'light') {
                document.documentElement.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                themeToggle.classList.remove('light');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
                themeToggle.classList.add('light');
            }
        });
        
        navbar.appendChild(themeToggle);
    }
}

// 增强数据可视化
function enhanceVisualizations() {
    // 添加动态波浪效果到图表背景
    const chartContainers = document.querySelectorAll('.chart-container');
    
    chartContainers.forEach(container => {
        const wave = document.createElement('div');
        wave.className = 'wave-container';
        wave.innerHTML = `
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
        `;
        
        container.appendChild(wave);
    });
    
    // 图表响应鼠标悬停效果
    const chartCanvases = document.querySelectorAll('.chart-canvas');
    
    chartCanvases.forEach(canvas => {
        canvas.addEventListener('mouseover', function() {
            this.style.transform = 'scale(1.02)';
        });
        
        canvas.addEventListener('mouseout', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

// 设置监控面板实时更新
function setupMonitoringPanel() {
    const monitoringPanel = document.querySelector('.monitoring-panel');
    if (!monitoringPanel) return;
    
    // 示例数据（实际使用时会从后端API获取）
    let monitoringData = {
        activeTopics: 32,
        warningTopics: 8,
        interventionTopics: 3,
        processingRate: 98.5,
        lastUpdate: new Date()
    };
    
    // 更新监控数据
    function updateMonitoring() {
        // 在实际应用中，这里会调用API获取最新数据
        
        // 模拟数据变化
        monitoringData.activeTopics += Math.floor(Math.random() * 3) - 1;
        monitoringData.warningTopics += Math.floor(Math.random() * 3) - 1;
        monitoringData.interventionTopics = Math.max(0, monitoringData.interventionTopics + Math.floor(Math.random() * 3) - 1);
        monitoringData.processingRate = (Math.random() * 2 + 97).toFixed(1);
        monitoringData.lastUpdate = new Date();
        
        // 更新UI
        document.getElementById('active-topics').textContent = monitoringData.activeTopics;
        document.getElementById('warning-topics').textContent = monitoringData.warningTopics;
        document.getElementById('intervention-topics').textContent = monitoringData.interventionTopics;
        document.getElementById('processing-rate').textContent = monitoringData.processingRate + '%';
        document.getElementById('last-update').textContent = formatDate(monitoringData.lastUpdate);
    }
    
    // 定期更新
    if (monitoringPanel) {
        // 初始更新
        updateMonitoring();
        
        // 每30秒更新一次
        setInterval(updateMonitoring, 30000);
    }
}

// 首页动画效果
function setupHomeAnimations() {
    // 添加进入视图动画
    const animatedElements = document.querySelectorAll('.glass-card, .result-card');
    
    // 检测元素是否在视口中
    function isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.bottom >= 0
        );
    }
    
    // 初始设置
    animatedElements.forEach(element => {
        // 初始不可见
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    });
    
    // 滚动时检查元素
    function checkElements() {
        animatedElements.forEach(element => {
            if (isInViewport(element) && element.style.opacity === '0') {
                setTimeout(() => {
                    element.style.opacity = '1';
                    element.style.transform = 'translateY(0)';
                }, 100); // 添加一点延迟，使动画看起来更自然
            }
        });
    }
    
    // 初始检查
    window.addEventListener('load', checkElements);
    
    // 滚动时检查
    window.addEventListener('scroll', checkElements);
}

// 格式化日期
function formatDate(date) {
    const now = new Date();
    const diff = Math.abs(now - date) / 1000; // 差异（秒）
    
    if (diff < 60) {
        return '刚刚';
    } else if (diff < 3600) {
        const minutes = Math.floor(diff / 60);
        return `${minutes}分钟前`;
    } else if (diff < 86400) {
        const hours = Math.floor(diff / 3600);
        return `${hours}小时前`;
    } else {
        // 自定义日期格式
        return `${date.getFullYear()}-${('0' + (date.getMonth() + 1)).slice(-2)}-${('0' + date.getDate()).slice(-2)} ${('0' + date.getHours()).slice(-2)}:${('0' + date.getMinutes()).slice(-2)}`;
    }
}

// 创建和显示仪表盘
function createGauge(element, value, label) {
    const gaugeContainer = document.createElement('div');
    gaugeContainer.className = 'gauge-container';
    
    const percentValue = Math.min(Math.max(value, 0), 100);
    const rotationDegree = -90 + (percentValue * 1.8); // -90到90度的范围
    
    gaugeContainer.innerHTML = `
        <div class="gauge-background">
            <div class="gauge-fill" style="height: ${percentValue}%"></div>
            <div class="gauge-ticks">
                ${createGaugeTicks()}
            </div>
        </div>
        <div class="gauge-center"></div>
        <div class="gauge-needle" style="transform: rotate(${rotationDegree}deg)"></div>
        <div class="gauge-value">${percentValue.toFixed(1)}%</div>
        <div class="gauge-label">${label}</div>
    `;
    
    element.appendChild(gaugeContainer);
    
    // 触发动画
    setTimeout(() => {
        const gaugeFill = gaugeContainer.querySelector('.gauge-fill');
        const gaugeNeedle = gaugeContainer.querySelector('.gauge-needle');
        
        gaugeFill.style.height = percentValue + '%';
        gaugeNeedle.style.transform = `rotate(${rotationDegree}deg)`;
    }, 100);
}

// 创建仪表盘刻度
function createGaugeTicks() {
    let ticks = '';
    for (let i = 0; i <= 10; i++) {
        const rotation = -90 + (i * 18); // -90到90度均匀分布
        const length = i % 5 === 0 ? 15 : 10; // 主刻度线更长
        
        ticks += `<div class="gauge-tick" style="transform: rotate(${rotation}deg) translateY(-${length}px);"></div>`;
    }
    return ticks;
}

// 创建和显示统计摘要
function createStatsSummary(container, stats) {
    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'stats-summary';
    
    for (const [key, value] of Object.entries(stats)) {
        const statCard = document.createElement('div');
        let className = 'stat-card';
        
        // 根据值确定颜色
        if (value.type === 'percentage') {
            const percent = parseFloat(value.value);
            if (percent > 70) className += ' high';
            else if (percent > 40) className += ' medium';
            else className += ' low';
        } else if (value.level) {
            className += ` ${value.level}`;
        }
        
        statCard.className = className;
        statCard.innerHTML = `
            <div class="stat-value">${value.value}</div>
            <div class="stat-label">${value.label}</div>
        `;
        
        summaryDiv.appendChild(statCard);
    }
    
    container.appendChild(summaryDiv);
}

// 鼠标悬停工具提示
function setupTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    
    tooltips.forEach(element => {
        const tooltipText = element.getAttribute('data-tooltip');
        
        // 创建工具提示容器
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        
        // 复制原始元素内容
        tooltip.innerHTML = element.innerHTML + `<span class="tooltip-text">${tooltipText}</span>`;
        
        // 替换原始元素
        element.parentNode.replaceChild(tooltip, element);
    });
}

// 添加响应式数据表格
function createResponsiveTable(container, headers, data) {
    const table = document.createElement('div');
    table.className = 'responsive-table';
    
    // 创建表头
    const tableHeader = document.createElement('div');
    tableHeader.className = 'table-header';
    
    headers.forEach(header => {
        const headerCell = document.createElement('div');
        headerCell.className = 'table-cell';
        headerCell.textContent = header.label;
        if (header.width) headerCell.style.width = header.width;
        tableHeader.appendChild(headerCell);
    });
    
    table.appendChild(tableHeader);
    
    // 创建表格内容
    data.forEach(row => {
        const tableRow = document.createElement('div');
        tableRow.className = 'table-row';
        
        headers.forEach(header => {
            const cell = document.createElement('div');
            cell.className = 'table-cell';
            
            // 检查是否有自定义渲染函数
            if (header.render) {
                cell.innerHTML = header.render(row[header.key], row);
            } else {
                cell.textContent = row[header.key] || '-';
            }
            
            tableRow.appendChild(cell);
        });
        
        table.appendChild(tableRow);
    });
    
    container.appendChild(table);
}

// 自定义选择器
function setupCustomSelects() {
    const customSelects = document.querySelectorAll('.custom-select');
    
    customSelects.forEach(select => {
        const selectElement = select.querySelector('select');
        if (!selectElement) return;
        
        // 创建自定义选择器
        const customSelectDiv = document.createElement('div');
        customSelectDiv.className = 'custom-select';
        
        // 当前选中项
        const selectedOption = document.createElement('div');
        selectedOption.className = 'select-selected';
        selectedOption.textContent = selectElement.options[selectElement.selectedIndex].text;
        
        // 选项列表
        const optionsList = document.createElement('div');
        optionsList.className = 'select-items select-hide';
        
        Array.from(selectElement.options).forEach((option, index) => {
            const optionItem = document.createElement('div');
            optionItem.className = 'select-item';
            optionItem.textContent = option.text;
            optionItem.dataset.value = option.value;
            optionItem.dataset.index = index;
            
            optionItem.addEventListener('click', function() {
                // 更新选中值
                selectElement.selectedIndex = this.dataset.index;
                selectedOption.textContent = this.textContent;
                
                // 触发change事件
                const event = new Event('change');
                selectElement.dispatchEvent(event);
                
                // 隐藏选项列表
                optionsList.classList.add('select-hide');
            });
            
            optionsList.appendChild(optionItem);
        });
        
        // 点击显示/隐藏选项列表
        selectedOption.addEventListener('click', function(e) {
            e.stopPropagation();
            closeAllSelect(this);
            optionsList.classList.toggle('select-hide');
            this.classList.toggle('select-arrow-active');
        });
        
        // 添加元素
        customSelectDiv.appendChild(selectedOption);
        customSelectDiv.appendChild(optionsList);
        
        // 替换原始选择器
        select.parentNode.replaceChild(customSelectDiv, select);
    });
    
    // 点击其他地方关闭所有选择框
    document.addEventListener('click', closeAllSelect);
}

function closeAllSelect(elmnt) {
    const selectItems = document.getElementsByClassName('select-items');
    const selectSelected = document.getElementsByClassName('select-selected');
    
    Array.from(selectItems).forEach((items, idx) => {
        if (elmnt !== selectSelected[idx]) {
            items.classList.add('select-hide');
            if (selectSelected[idx]) {
                selectSelected[idx].classList.remove('select-arrow-active');
            }
        }
    });
}