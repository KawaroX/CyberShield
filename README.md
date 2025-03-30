# CyberShield - 网络暴力预警和治理系统

CyberShield 是一个基于自然语言处理和机器学习技术的网络暴力预警和治理系统，旨在自动检测、预警和应对网络空间中的暴力内容。

## 功能特点

- **内容分析**：分析单条内容的情感极性和暴力倾向
- **上下文感知**：能够结合上下文理解内容的真实含义
- **话题聚类**：将相关内容聚合成话题，识别潜在风险
- **风险评估**：评估内容和话题的暴力风险程度
- **干预建议**：根据风险程度生成干预策略

## 技术架构

- **前端**：HTML, CSS, JavaScript, Chart.js
- **后端**：Python, Flask
- **数据存储**：MongoDB
- **NLP**：百度AI开放平台, 嵌入模型(BGE)
- **机器学习**：scikit-learn, numpy

## 安装与使用

### 环境要求

- Python 3.6+
- MongoDB
- Ollama (用于BGE模型)

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/KawaroX/CyberShield.git
   cd CyberShield
   ```
2.创建并激活虚拟环境
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 或 venv\Scripts\activate  # Windows
    ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
4. 配置环境变量
   ```bash
   export BAIDU_API_KEY=你的百度API密钥
   export BAIDU_SECRET_KEY=你的百度密钥
   ```
5. 启动服务
   ```bash
   python run.py
   ```

## 使用方法

访问 http://localhost:5001/ 使用系统。

## 贡献指南

欢迎贡献！请提交 Issue 或 Pull Request。

## 许可证

本项目采用 MIT 许可证。