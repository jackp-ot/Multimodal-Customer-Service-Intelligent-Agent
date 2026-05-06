# AI Agent RAG - 多模态智能问答系统

基于 **ReAct Agent + 多模态 RAG** 的智能问答系统，支持产品手册的自动化知识检索与图文融合回答。

## 项目概述

我构建了一个以 **ReAct Agent** 为核心的智能问答工作流，并在此基础上搭建了**多模态 RAG 检索增强生成系统**，通过混合检索与多模型协同实现高质量的知识问答，整体形成「文档解析 → 向量索引 → 混合检索 → 多模态生成」的闭环。

工作流中，任务拆解为多组件协同：**ReactAgent** 负责意图理解与工具调度，**RagSummaryService** 负责多模态检索与内容总结，**Middleware** 层负责工具调用监控与日志追踪。Agent 通过 `rag_retrieve_tool` 自动调用向量库检索相关文档片段，结合 System Prompt 进行多轮推理与自反馈，能自动完成从问题理解、知识检索到答案生成的完整流程。

RAG 系统侧，核心是围绕「混合检索 + 多模态理解」的检索增强架构。系统基于 **Milvus** 向量数据库，采用**稠密向量（DashScope Embedding）+ 稀疏向量（BM25）**双路混合检索，并通过 **RRF 重排序**提升召回精度。文档侧支持 TXT/PDF 多格式解析，结合 **TxtSmartChunker** 智能分块算法，按段落与句子边界自适应切割，保留 `<PIC>` 图片标记与元数据的映射关系。多模态层支持**通义千问 VL** 视觉模型，自动加载产品插图并以 base64 注入上下文，实现图文融合的增强回答。

## 项目结构

```
Agent/
├── agent/                    # Agent 核心
│   ├── react_agent.py        # ReAct Agent 主逻辑
│   └── tools/
│       ├── agent_tools.py    # RAG 检索工具定义
│       └── middle_ware.py    # 工具调用监控与日志中间件
├── config/                   # 配置文件
│   ├── agent.example.yml     # Agent 模型配置模板
│   ├── milvus.example.yml    # Milvus 数据库配置模板
│   ├── prompt.yml            # Prompt 路径配置
│   └── rag.example.yml       # RAG 模型配置模板
├── data/                     # 知识库数据
│   ├── 插图/                  # 产品插图
│   └── *.txt                 # 产品手册（TXT/PDF）
├── model/                    # 模型工厂
│   └── factory.py            # LLM / Embedding / 多模态模型初始化
├── prompts/                  # Prompt 模板
│   ├── main_prompt.txt       # Agent 系统提示词
│   ├── rag_summary.txt       # RAG 总结提示词
│   └── report.txt            # 报告生成提示词
├── rag/                      # RAG 检索增强
│   ├── milvus_db_dense.py    # Milvus 向量数据库（稠密+稀疏混合检索）
│   ├── rag_service.py        # 多模态 RAG 服务
│   ├── txt_chunk.py          # TXT 智能分块器
│   └── vector_store.py       # 向量存储服务
├── scripts/                  # 脚本工具
│   ├── batch_qa.py           # 批量问答脚本
│   └── reload_db.py          # 知识库重载脚本
├── utils/                    # 工具模块
│   ├── config_handler.py     # 配置加载
│   ├── file_handler.py       # 文件处理（MD5/PDF/TXT）
│   ├── logger_handler.py     # 日志管理
│   ├── path_tool.py          # 路径工具
│   └── prompt_load.py        # Prompt 加载
├── web/                      # Web 服务
│   ├── main.py               # FastAPI 应用
│   ├── static/               # 静态资源
│   └── templates/            # HTML 模板
├── .env.example              # 环境变量模板
├── .gitignore
├── pyproject.toml            # uv 项目配置
└── README.md
```

## 快速开始

### 环境要求

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv)（推荐的 Python 包管理器）
- Milvus 向量数据库（本地或远程）

### 1. 安装 uv

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆项目

```bash
git clone https://github.com/your-username/ai-agent-rag.git
cd ai-agent-rag
```

### 3. 配置环境

```bash
# 复制配置模板
copy config\rag.example.yml config\rag.yml
copy config\agent.example.yml config\agent.yml
copy config\milvus.example.yml config\milvus.yml

# 编辑配置文件，填入你的 API Key 和 Milvus 连接信息
# config\rag.yml      - 填入阿里云 DashScope API Key
# config\milvus.yml   - 填入 Milvus 数据库地址和端口
```

### 4. 安装依赖

```bash
uv sync
```

### 5. 加载知识库

将产品手册（TXT/PDF）放入 `data/` 目录，然后运行：

```bash
uv run python scripts/reload_db.py
```

### 6. 启动 Web 服务

```bash
uv run uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 [http://localhost:8000](http://localhost:8000) 即可使用 Web 聊天界面。

## 使用方式

### Web 聊天界面

启动 Web 服务后，在浏览器中打开，支持：
- 纯文本问答（自动 RAG 检索）
- 图片上传 + 视觉分析
- 多轮对话管理
- SSE 流式输出（打字机效果）

### 批量问答

```bash
uv run python scripts/batch_qa.py
```

从 CSV 文件批量读取问题，自动调用 Agent 回答并输出结果。

### 命令行测试

```bash
uv run python agent/react_agent.py
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 聊天界面 |
| POST | `/api/chat` | 发送消息（SSE 流式） |
| POST | `/api/conversations` | 创建新对话 |
| GET | `/api/conversations` | 获取对话列表 |
| GET | `/api/conversations/{id}` | 获取对话详情 |
| DELETE | `/api/conversations/{id}` | 删除对话 |
| GET | `/api/images/{name}` | 获取插图 |

## 技术架构

```
用户提问
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│  FastAPI     │────▶│   ReactAgent     │
│  Web 服务    │     │   (LangChain)    │
└─────────────┘     └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  rag_retrieve    │
                    │  _tool           │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌──────────────┐
     │ 稠密检索    │  │ 稀疏检索    │  │ 多模态模型    │
     │ (Embedding)│  │ (BM25)     │  │ (Qwen-VL)    │
     └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
           │               │                │
           └───────┬───────┘                │
                   │                        │
                   ▼                        │
          ┌──────────────┐                  │
          │ RRF 重排序    │                  │
          │ (Milvus)     │                  │
          └──────┬───────┘                  │
                 │                          │
                 └──────────┬───────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  图文融合回答   │
                   │  (带<PIC>标记) │
                   └────────────────┘
```

### 核心技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangChain / LangGraph |
| 大语言模型 | DeepSeek-V4 (阿里云 DashScope) |
| 视觉模型 | 通义千问 VL (Qwen-VL-Plus) |
| Embedding | DashScope text-embedding-v4 |
| 向量数据库 | Milvus (稠密 + 稀疏混合检索) |
| 稀疏检索 | BM25 (pymilvus BM25EmbeddingFunction) |
| 重排序 | RRF (Reciprocal Rank Fusion) |
| Web 框架 | FastAPI + SSE 流式输出 |
| 文档解析 | PyPDF + 自定义 TXT 智能分块 |

## 配置说明

### rag.yml / agent.yml

```yaml
chat_model_name: deepseek-v4-flash      # 对话模型
vision_model_name: qwen-vl-plus          # 多模态视觉模型
embedding_model_name: text-embedding-v4  # Embedding 模型
api_key: your_api_key_here               # 阿里云 DashScope API Key
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

### milvus.yml

```yaml
data_path: data                    # 知识库数据目录
allowed_file_type: ["txt", "pdf"]  # 允许的文件类型
collection_name: QA_agent          # Milvus 集合名称
k: 5                               # 默认检索数量
host: 127.0.0.1                    # Milvus 地址
port: 19530                        # Milvus 端口
token: ""                          # Milvus 认证 Token
```

## 数据格式

### TXT 产品手册（多模态格式）

支持 JSON 格式的多模态文本，包含 `<PIC>` 标记与图片名称的映射：

```json
["产品介绍文本<PIC>详细说明<PIC>", ["image_01", "image_02"]]
```

图片文件放入 `data/插图/` 目录，支持 JPG/PNG 格式。

## License

MIT
