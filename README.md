# CyberShield - 网络暴力预警和治理系统

![CyberShield Logo](image.png)

CyberShield 是一个基于自然语言处理和机器学习技术的网络暴力预警和治理系统，旨在自动检测、预警和应对网络空间中的暴力内容。系统通过聚类和多维度分析用户生成内容，实现对网络暴力的早期识别和智能干预。

## 系统概述

CyberShield 采用双层分析架构，结合微观内容分析和宏观话题追踪，为网络平台提供全方位的安全保障：

### 预警系统

- **内容采集**：收集网络评论、社交媒体帖子等用户生成内容
- **话题聚类**：使用 BGE 嵌入模型对内容进行聚类，识别热点话题
- **情绪分析**：利用百度 NLP API 分析内容情感极性和情绪特征
- **暴力检测**：专门设计的暴力词典和检测算法，区分负面评价与网络暴力
- **风险评估**：对话题进行综合风险评分，超过阈值触发预警

### 治理系统

- **内容审核**：对预警话题相关内容进行实时审核
- **上下文感知**：考虑内容的上下文环境，避免误判
- **分级干预**：根据暴力程度采取不同级别的干预措施
  - 低风险：正常发布
  - 中风险：警告用户但允许发布
  - 高风险：人工审核
  - 极高风险：自动拒绝发布
- **干预建议**：为平台管理者提供有针对性的治理建议

## 技术特点

- **双层分析框架**：结合微观内容分析和宏观话题分析
- **上下文感知**：能够结合上下文理解内容的真实含义
- **实时聚类**：动态识别和追踪潜在风险话题
- **多维度风险评估**：综合考虑暴力程度、情感极性、传播速度等因素
- **适应性干预**：根据不同风险级别和话题特征生成定制化干预策略

## 技术架构

### 前端

- HTML5, CSS3, JavaScript
- Chart.js 可视化库
- 响应式设计，支持多设备访问

### 后端

- **Web 框架**：Python Flask
- **数据存储**：MongoDB
- **NLP 技术**：
  - 百度 AI 开放平台（情感分析、关键词提取）
  - 嵌入模型：BGE (BAAI General Embedding)
  - 自定义暴力检测算法
- **机器学习**：scikit-learn (聚类和分类)

### 系统组件

- **暴力检测器 (ViolenceDetector)**：基于自定义词典和规则的暴力内容识别
- **话题管理器 (TopicManager)**：负责内容聚类和话题追踪
- **内容分析模型 (ContentAnalysis)**：微观层面的内容特征分析
- **话题分析模型 (TopicAnalysis)**：宏观层面的话题风险评估

## 安装与配置

### 环境要求

- Python 3.6+
- MongoDB
- Ollama (用于 BGE 模型)
- 百度 AI 开放平台账号

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/YourUsername/CyberShield.git
   cd CyberShield
   ```

2. 创建并激活虚拟环境
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
   # Linux/Mac
   export BAIDU_API_KEY=你的百度API密钥
   export BAIDU_SECRET_KEY=你的百度密钥
   
   # Windows
   set BAIDU_API_KEY=你的百度API密钥
   set BAIDU_SECRET_KEY=你的百度密钥
   ```

5. 配置 MongoDB（确保 MongoDB 服务已启动）

6. 配置 Ollama（对于 BGE 嵌入模型）
   ```bash
   # 安装 Ollama 并拉取 BGE 模型
   ollama pull bge-m3
   ```

7. 启动服务
   ```bash
   python run.py
   ```

## 使用指南

### Web 界面

访问 http://localhost:5001/ 使用系统的 Web 界面，包括以下功能：

- **内容分析**：输入文本进行暴力风险分析
- **话题监控**：查看和管理已识别的风险话题
- **上下文分析**：添加上下文内容进行综合分析
- **干预建议**：获取针对高风险话题的干预策略

### API 接口

CyberShield 提供以下主要 API 接口：

#### 内容分析接口

```
POST /api/analyze
Content-Type: application/json

{
    "content": "要分析的文本内容",
    "content_type": "text",
    "topic_id": "可选指定话题ID"
}
```

#### 上下文分析接口

```
POST /api/analyze_with_context
Content-Type: application/json

{
    "content": "要分析的文本内容",
    "content_type": "text",
    "context": [
        {
            "content": "上下文内容1",
            "content_id": "content-12345",
            "timestamp": "2023-01-01T00:00:00Z"
        }
    ]
}
```

#### 话题查询接口

```
GET /api/topics
```

#### 话题内容查询接口

```
GET /api/topics/{topic_id}/contents
```

#### 相似内容查询接口

```
POST /api/similar
Content-Type: application/json

{
    "text": "查询文本",
    "top_n": 5
}
```

## 自定义与扩展

### 自定义暴力词典

可以通过创建或修改暴力词典文件来自定义暴力检测规则：

```python
# 示例：更新暴力词典
from app.utils.violence_detector import ViolenceDetector

detector = ViolenceDetector()
detector.update_violence_dict({
    "新暴力词1": 0.5,
    "新暴力词2": 0.7
})
detector.save_violence_dict("custom_violence_dict.json")
```

### 自定义聚类参数

可以在 `config.py` 中调整聚类参数：

```python
# 聚类配置示例
NUM_CLUSTERS = 10  # 聚类数量
VIOLENCE_THRESHOLD = 0.7  # 暴力内容阈值
EARLY_WARNING_THRESHOLD = 0.5  # 预警阈值
```

### 扩展分析维度

系统设计支持添加新的分析维度，例如：

1. 在 `ContentAnalysis` 类中添加新的特征字段
2. 实现新的特征提取函数
3. 更新风险评估和干预策略

## 未来发展计划

### 模型升级

- **网络暴力专用模型**：收集网络暴力样本，训练专门的暴力检测模型
- **多语言支持**：扩展到更多语言的暴力内容检测
- **多模态分析**：支持图像、视频等多模态内容的暴力检测

### 功能增强

- **传播路径分析**：追踪网络暴力内容的传播路径和影响范围
- **用户画像**：分析参与网络暴力的用户特征和行为模式
- **干预效果评估**：监测和评估干预措施的实际效果

### 系统优化

- **性能优化**：提高系统处理大规模数据的能力
- **分布式部署**：支持分布式环境下的部署和扩展
- **实时监控**：增强实时监控和预警能力

## 贡献指南

欢迎贡献代码、提交 Issue 或改进建议！

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者：[Kawaro](mailto:wkawaro@gmail.com)
- 项目主页：[GitHub Repository](https://github.com/KawaroX/CyberShield)

## 致谢

- 感谢百度 AI 开放平台提供的 NLP 服务
- 感谢 BAAI 提供的 BGE 嵌入模型
- 感谢所有对本项目做出贡献的开发者

---

*CyberShield - 为健康网络环境保驾护航*