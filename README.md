# DeepAgents Search

基于 DeepAgents 框架的多智能体深度搜索系统。1 个主 Agent 协调 3 个专家子 Agent，覆盖网络搜索、结构化数据库查询、私有知识库检索三大场景，通过 BERT 路由 + LangGraph 流水线 + WebSocket 实时监控，实现复杂查询任务的智能分发与闭环执行。

## 架构概览

```
用户请求 → FastAPI → 主 Agent (DeepAgents)
                       ├─ 对话记忆 (多轮上下文 + 指代消解)
                       ├─ BERT 路由 → 子 Agent 智能分发 + Query 拆解
                       ├─ 网络搜索助手 (多角度深度搜索)
                       ├─ 数据库查询助手 (LangGraph 12 节点流水线)
                       │    └─ Qdrant 向量检索 + ES 全文检索 + MySQL 执行
                       ├─ 知识库助手 (本地 RAG 管线)
                       │    └─ HyDE → BGE-M3 → Qdrant → BGE Reranker
                       └─ 工具层: Markdown 生成 / PDF 导出 / 文件读写
```

## 技术方案详解

### 1. BERT 子代理路由

传统方案完全依赖 LLM 做子代理路由，存在幻觉风险：LLM 可能将"查询库存"误路由到网络搜索。本项目采用 **BERT + LLM 混合路由**，BERT 负责意图分类，LLM 负责执行编排，两者互补。

**路由内部流程**：

```
用户Query
    │
    ▼
┌─────────────────┐
│  LLM 指代消解     │  ← 短查询/有历史时，补全省略的主语
│  "那销售额呢？"    │     结合对话记忆做指代消解
│  → "布洛芬销售额"  │
└────────┬────────┘
         │  完整 Query
         ▼
┌─────────────────┐
│  BERT 推理       │  ← 单次 forward pass，3 个标签同时计算
│  sigmoid 独立判断 │     [db_query: 0.87, web_search: 0.79, kb_search: 0.12]
└────────┬────────┘
         │  三段式判定
         ▼
    ┌────────┬────────┬────────┐
    │        │        │        │
 全部<0.35  部分0.35  任一≥0.5
    │      ~0.5之间   │
    ▼        ▼        ▼
 ┌──────┐ ┌──────┐ ┌──────────┐
 │ none │ │ LLM  │ │  bert    │
 │ 无关 │ │兜底  │ │  命中    │
 └──┬───┘ └──┬───┘ └────┬─────┘
    │        │          │
    ▼        ▼          ▼
 告知主Agent 不给建议  命中≥2个?
 "直接回答"  LLM自己判    │
                    ┌────┴────┐
                    │ 是       │ 否
                    ▼          ▼
              ┌──────────┐  ┌──────────┐
              │Query拆解  │  │ 直接注入  │
              │LLM拆为子任务│  │ 路由建议  │
              └────┬─────┘  └────┬─────┘
                   │             │
                   ▼             │
              ┌──────────┐       │
              │实体覆盖校验│       │
              │(覆盖率≥50%)│      │
              └────┬─────┘       │
                   │             │
              ┌────┴────┐        │
              │通过  │失败│        │
              ▼      ▼    │      │
            注入   回退原   │      │
           子任务  Query   │      │
              │      │    │      │
              └──────┴────┴──────┘
                     │
                     ▼
             注入 route_hint 到主 Agent prompt
```

**关键设计点**：

- **模型**: `hfl/chinese-roberta-wwm-ext`（哈工大讯飞联合实验室，全词掩码预训练，110M 参数）。单次推理一次 forward pass 同时产出 3 个标签分数，非三次推理
- **多标签 sigmoid**: 每个标签独立判断，分数不求和为 1。`0.87 + 0.79 + 0.12 = 1.78` 是对的——说明 query 同时高度命中两个子代理。对比 softmax 会压制弱标签，导致"查库存+搜行情"只命中一个
- **三段式路由决策**: 全部 < 0.35 → 判定为无关闲聊，显式阻止子代理调用；0.35~0.5 → LLM 自主判断；≥ 0.5 → BERT 路由
- **Query 拆解**: 命中多个子代理时，LLM 将原 query 拆解为各子代理的精准子任务（如 `db_query: "查询布洛芬库存"`, `web_search: "搜索布洛芬市场行情"`），子代理各自收到降噪后的描述，而非完整原 query
- **防幻觉三层防御**: ① Prompt 严格约束"只拆不增，禁止编造"；② jieba TF-IDF 实体覆盖率校验（合并子任务后关键实体需保留原 query 的 50%+，编造实体 < 50%）；③ 校验失败回退传原 query，零风险
- **已知局限 — 语义串台**: LLM 拆解时可能将 A 意图的实体错配给 B 子代理（如"布洛芬库存"分配给 web_search、"阿莫西林政策"分配给 db_query）。合并实体覆盖率检查无法发现（合并后实体仍完整，但分配交叉了）。这是 NLU 开放域语义理解问题，无可靠自动校验手段，依赖主 Agent LLM 在调用 `task()` 时根据原始 query 自行纠正
- **微调**: 内置 `router/trainer.py`，支持单标签/多标签混合标注，BCEWithLogitsLoss 训练

### 2. 对话记忆

按 `thread_id` 隔离的多轮对话记忆，支撑指代消解和上下文连贯：

- **短期记忆**: 每会话保留最近 20 轮对话，自动裁剪旧记录
- **上下文注入**: 新一轮查询时，将最近 5 轮历史格式化为自然语言注入 prompt
- **指代消解**: 路由模块加载对话历史，将 `"那销售额呢？"` 消解为 `"布洛芬的销售额是多少？"`
- **线程安全**: asyncio.Lock 保护每会话写操作
- **存储**: 当前为进程内存，可替换为 SQLite/Redis 持久化后端

### 3. 网络搜索助手

多角度深度搜索管线，通过查询改写 + 硬编码多轮并行搜索 + 去重聚合，构建高密度全局上下文：

```
用户Query → LLM 改写 (生成 3~5 个不同角度搜索词)
→ ThreadPoolExecutor 并行执行所有搜索 (硬编码规则，不可跳过)
→ URL 精确去重 + 结果聚合 → 高密度搜索报告
```

| 环节 | 技术 | 说明 |
|---|---|---|
| 查询改写 | LLM (qwen-max) | 从定义/原理/应用/案例/趋势等多维度拆解，生成 3~5 个搜索词 |
| 并行搜索 | Tavily Search API + ThreadPoolExecutor | 硬编码强制执行所有角度搜索，advanced 深度模式 |
| 去重聚合 | URL 精确去重 | 消除跨角度重复结果，保留最多 20 条交错的全局视角 |
| 上下文构建 | 结构化报告格式 | 含搜索角度清单 + 来源标注 + 内容摘要 |

核心设计理念：通过多角度、多关键词的广域搜索，有效降低单一信息源偏见，避免"信息茧房"效应。

### 3. 数据库查询助手（SQL Agent）

这是整个系统中技术密度最高的模块，将自然语言问题自动转换为可执行的 SQL 查询。内部是一条 **LangGraph StateGraph 12 节点流水线**：

```
关键词提取 → 三路并行召回（列向量/指标向量/值全文）
→ 检索结果合并 → LLM 过滤表/指标 → 补充上下文
→ SQL 生成 → EXPLAIN 校验 → (正确则执行, 错误则 LLM 纠错后执行)
```

各环节技术选型：

| 环节 | 技术 | 说明 |
|---|---|---|
| 关键词提取 | jieba 分词 + TF-IDF | 中文分词后取 TopK 关键词 |
| 列名召回 | BGE-large-zh-v1.5 + Qdrant | 字段描述向量化，语义匹配用户意图 |
| 指标召回 | BGE-large-zh-v1.5 + Qdrant | 指标定义向量化，如"销售额"→total_amount |
| 字段值召回 | Elasticsearch + ik_max_word | 字段取值全文索引，如"布洛芬"→drugs.generic_name |
| 表/指标过滤 | LLM (qwen-max) | 从召回的候选表中筛选真正相关的表和指标 |
| SQL 生成 | LLM (qwen-max) | 基于过滤后的上下文 + 用户问题生成 SQL |
| SQL 校验 | MySQL EXPLAIN | 语法校验，失败则 LLM 纠错重试 |
| 元数据管理 | MySQL meta 库 | 表/列/指标的描述和别名集中存储 |

**Repository 模式**：Entity（dataclass）+ ORM Model（SQLAlchemy）+ Repository + Mapper，领域层与持久层解耦。

### 4. 知识库助手（RAG Agent）

本地私有知识库检索管线，不依赖任何外部 RAG 服务，数据不出域：

```
用户问题 → HyDE 假设文档生成 → BGE-M3 向量化
→ Qdrant 子块检索 (Top20) → 父块追溯
→ BGE Reranker v2-m3 Cross-Encoder 精排 → 返回 TopK
```

各环节技术选型：

| 环节 | 技术 | 说明 |
|---|---|---|
| 文档切分 | 父子文档关联切分 | Markdown 标题优先，子块 512 字符 / 父块 2048 字符 |
| 查询增强 | HyDE (Hypothetical Document Embedding) | LLM 生成假设答案，与原始问题拼接，增强检索语义 |
| 向量化 | BGE-M3 (1024 维) | FlagEmbedding 原生加载，支持稠密+稀疏双路输出 |
| 向量存储 | Qdrant 父子双集合 | 子块集合用于精准匹配，父块集合返回完整上下文 |
| 重排序 | BGE Reranker v2-m3 | Cross-Encoder 精排，动态 TopK（断崖检测：相对差 >25% 或绝对差 >0.5 时截断） |
| 去重 | MD5 文件指纹 | 增量构建，仅处理新增/变更文件 |

### 6. 前端

Vue 3 + Vite 单页应用，Composition API + `<script setup>` 语法：

- **组件结构**: Sidebar（子代理列表 / 实时事件 / 文件管理）、ChatPanel（消息气泡 + 思考过程时间线）、InputBar（查询输入 + 文件上传 Chip）
- **状态管理**: `provide/inject` 跨组件共享 threadId、消息列表、WebSocket 状态
- **WebSocket**: 自动连接 + 心跳保活 + 断线重连，实时渲染 Agent 思考过程（路由决策 → 调用助手 → 工具执行 → 最终结果）
- **构建产物**: `ui/dist/` 纯静态文件，FastAPI 自动托管，也可单独部署到任何 Web 服务器

## 项目结构

```
Deepagent_search/
├── agent/                     # 主 Agent + 3 个子 Agent
│   ├── main_agent.py          # Agent 组装、环境准备、流式执行
│   ├── llm.py                 # LLM 初始化
│   ├── prompts.py             # 提示词加载
│   └── sub_agents/
│       ├── network_search_agent.py    # 网络搜索助手
│       ├── database_query_agent.py    # 数据库查询助手 (CompiledSubAgent)
│       └── knowledge_base_agent.py    # 知识库助手 (本地 RAG)
├── sql_agent/                 # SQL 生成子 Agent (LangGraph 流水线)
│   ├── agent/                 # 12 节点 StateGraph + 状态/上下文定义
│   ├── evaluation/            # SQL 评测模块 (BIRD-style EX)
│   ├── clients/               # MySQL/Qdrant/ES/Embedding 连接管理
│   ├── entities/              # 领域实体 (dataclass)
│   ├── models/                # SQLAlchemy ORM 模型
│   ├── repositories/          # Repository 层 (mysql/qdrant/es)
│   ├── services/              # 元数据知识库构建服务
│   ├── scripts/               # CLI 入口 (build_meta_knowledge)
│   ├── conf/                  # OmegaConf 配置
│   └── prompt/                # SQL 生成 prompt 模板
├── rag_agent/                 # RAG 知识库子 Agent (本地检索管线)
│   ├── loader.py              # 文档加载 + MD5 去重
│   ├── splitter.py            # 父子文档关联切分
│   ├── embedder.py            # BGE-M3 向量化封装
│   ├── hyde.py                # HyDE 假设文档生成
│   ├── qdrant_store.py        # Qdrant 父子双集合存储
│   ├── reranker.py            # BGE Reranker 重排序 + 动态 TopK
│   ├── retriever.py           # 完整检索管线编排
│   ├── builder.py             # 知识库构建编排器
│   ├── config.py              # RAG 配置 (dataclass)
│   └── scripts/
│       ├── build_kb.py        # CLI 知识库构建
│       └── evaluate.py        # CLI RAG 评测
├── router/                    # BERT 子代理路由模块
│   ├── classifier.py          # BERT 意图分类器
│   ├── trainer.py             # 微调训练器
│   ├── integration.py         # 路由集成 (改写→分类→路由建议)
│   ├── config.py              # 路由配置
│   └── data/
│       └── annotation_template.json  # 标注数据模板
├── evaluation/                # 统一评测模块
│   ├── cli.py                 # 评测 CLI 入口
│   ├── router_eval.py         # BERT 路由评测
│   ├── rag_eval.py            # RAG 管线评测
│   └── sql_eval.py            # SQL Agent 评测
├── api/                       # FastAPI 服务层
│   ├── server.py              # REST + WebSocket 服务
│   ├── monitor.py             # 实时监控 + 连接管理
│   ├── context.py             # ContextVar 会话隔离
│   ├── memory.py              # 对话记忆管理器
│   └── logger.py              # 会话日志
├── tools/                     # 工具函数
│   ├── tavily_tools.py        # Tavily 客户端
│   ├── deep_search_tools.py   # 多角度深度搜索工具
│   ├── markdown_tools.py      # Markdown 生成
│   ├── pdf_tools.py           # MD → PDF 导出
│   └── upload_file_read_tool.py  # 多格式文件读取
├── conf/
│   ├── app_config.yaml        # 基础设施配置 (DB/Qdrant/ES/LLM)
│   └── meta_config.yaml       # 数据库元数据描述 (表/字段/指标)
├── prompt/
│   └── prompts.yml            # Agent 提示词配置
├── sql/
│   └── company_data.sql       # 示例数据库 (医药公司)
├── ui/
│   ├── index.html             # Vite 入口 HTML
│   ├── package.json           # Vue 3 + Vite 依赖
│   ├── vite.config.js         # Vite 配置 (含 API 代理)
│   └── src/
│       ├── main.js            # Vue 入口
│       ├── App.vue            # 根组件 (布局 + 全局样式)
│       ├── components/
│       │   ├── Sidebar.vue    # 侧栏 (子代理/事件/文件)
│       │   ├── ChatPanel.vue  # 聊天区 (消息 + 思考过程)
│       │   └── InputBar.vue   # 输入栏 (查询 + 上传)
│       └── composables/
│           ├── useWebSocket.js # WebSocket 连接
│           └── useApi.js       # REST API 封装
└── utils/                     # 通用工具
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- MySQL 8.0+
- Qdrant (端口 6333)
- Elasticsearch 7.x+ (端口 9200)

### 2. 安装依赖

```bash
pip install deepagents langgraph langchain langchain-core langchain-community \
    langchain-text-splitters fastapi uvicorn sqlalchemy asyncmy \
    qdrant-client elasticsearch transformers torch \
    FlagEmbedding sentence-transformers jieba pypdf python-docx \
    pyyaml omegaconf python-dotenv pydantic numpy
```

### 3. 启动基础设施

确保 MySQL、Qdrant、Elasticsearch 服务已启动，然后创建数据库：

```sql
CREATE DATABASE pharma_db;
CREATE DATABASE meta;
```

导入示例数据：

```bash
mysql -u root -p < sql/company_data.sql
```

### 4. 配置文件

编辑 `.env` 填入 API Key：

```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

编辑 `conf/app_config.yaml` 调整数据库连接信息（默认均为 localhost）。

### 5. 初始化知识库

**构建 SQL Agent 元数据知识库**（将表/字段/指标描述向量化写入 Qdrant + ES）：

```bash
python -m sql_agent.scripts.build_meta_knowledge
```

**构建 RAG 文档知识库**（将私有文档切分、向量化写入 Qdrant）：

```bash
# 先将文档放入 data/documents/ 目录
python -m rag_agent.scripts.build_kb
```

### 6. 统一评测

项目内置三项评测，覆盖路由 → 检索 → 生成的全链路质量评估。

**一键运行:**

```bash
# 全量评测
python -m evaluation.cli --all

# 快速模式 (少量样本试跑)
python -m evaluation.cli --all --quick

# 单项评测
python -m evaluation.cli --router     # BERT 路由评测
python -m evaluation.cli --rag        # RAG 管线评测
python -m evaluation.cli --sql        # SQL Agent 评测
```

#### 6.1 BERT 路由评测

指标: Accuracy (预测标签集合与真实标签集合完全一致的比例)

```bash
# 直接评测 (使用 annotation_template.json)
python -m evaluation.cli --router

# 指定数据
python -m evaluation.cli --router --router-data router/data/annotation_template.json
```

评测报告保存到 `data/router_eval_report.json`，同时列出错误案例（漏了谁/多判了谁）。

#### 6.2 RAG 管线评测

指标: RAGAS (Faithfulness / Answer Relevancy / Context Precision / Context Recall)

```bash
# 第一步: 生成测试集
python -m rag_agent.scripts.evaluate --generate --size 30

# 第二步: 评测
python -m evaluation.cli --rag

# 仅检索层 (省 token)
python -m evaluation.cli --rag --rag-retrieval-only
```

评测报告保存到 `data/rag_eval_report.json`。

| 指标 | 测什么 | 什么时候用 |
|---|---|---|
| Faithfulness | 回答有没有编造 | 调 prompt / 切 LLM 后 |
| Answer Relevancy | 答没答到点上 | 调生成参数后 |
| Context Precision | 检索结果里相关的多不多 | 调检索参数后 |
| Context Recall | 该检索的是不是都检了 | 调切分/向量化后 |

#### 6.3 SQL Agent 评测

指标: BIRD-style EX (执行准确率) + 错误分类

```bash
# 第一步: 生成测试集 (150 模板 + 50 LLM)
python -m sql_agent.evaluation.testset_generator --size 200

# 第二步: 评测
python -m evaluation.cli --sql

# 快速抽样
python -m evaluation.cli --sql --sql-limit 10
```

评测报告保存到 `data/sql_eval_report.json`。

| 指标 | 说明 |
|---|---|
| EX (Execution Accuracy) | 生成 SQL 执行结果 == 标准结果 |
| 按难度/类别分组 | simple / medium / hard, 聚合/关联/排序 等 |
| 错误分类 | 语法错误 / 语义错误(表列召错) / 空结果 / 超时 |

### 7. 微调 BERT 路由（可选）

扩充 `router/data/annotation_template.json` 中的标注数据后：

```bash
python -m router.trainer
```

微调后的模型自动保存到 `models/bert_router_finetuned/`，推理时优先加载。

### 8. 构建前端 + 启动服务

```bash
# 安装前端依赖 + 构建
cd ui && npm install && npm run build && cd ..

# 启动后端 (自动托管前端)
python api/server.py
```

服务启动在 `http://0.0.0.0:8000`。开发时也可分前后端各自启动：

```bash
# 终端 1: 后端
python api/server.py

# 终端 2: 前端 (热更新)
cd ui && npm run dev
# → http://localhost:5173 (自动代理 API 到 8000)
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/task` | 提交查询任务 `{query, thread_id?}` → `{status, thread_id}` |
| POST | `/api/upload` | 上传文件 `{files[], thread_id}` |
| GET | `/api/download?path=xxx` | 下载输出文件 |
| GET | `/api/files?path=xxx` | 列出会话目录文件 |
| WS | `/ws/{thread_id}` | WebSocket 实时监控推送 |

## 技术栈

| 层级 | 技术 |
|---|---|
| Agent 框架 | DeepAgents + LangGraph |
| LLM | 通义千问 (DashScope) |
| 路由 | BERT (hfl/chinese-roberta-wwm-ext) |
| 向量模型 | BGE-large-zh-v1.5 / BGE-M3 |
| 向量数据库 | Qdrant |
| 全文检索 | Elasticsearch |
| 重排序 | BGE Reranker v2-m3 |
| 关系数据库 | MySQL + SQLAlchemy async |
| 网络搜索 | Tavily Search API |
| Web 框架 | FastAPI + WebSocket |
| 前端 | Vue 3 + Vite (构建为纯静态文件) |

## 组件替换指南

每个组件都是可替换的，只需改配置或少量代码。

### 更换 LLM

在 `.env` 中修改：

```env
# 阿里云 DashScope → 其他兼容 OpenAI API 的服务
OPENAI_BASE_URL=https://api.openai.com/v1          # OpenAI
OPENAI_BASE_URL=https://api.deepseek.com/v1        # DeepSeek
OPENAI_BASE_URL=https://api.moonshot.cn/v1         # Moonshot
OPENAI_API_KEY=sk-your-key-here
LLM_QWEN_MAX=gpt-4o                                # 模型名改为目标模型
```

代码初始化在 `agent/llm.py`，用的是 `ChatOpenAI`，换任何兼容 OpenAI 接口的模型都行。

### 更换 BERT 路由模型

修改 `router/config.py`：

```python
model_name: str = "hfl/chinese-roberta-wwm-ext"   # → 换成其他中文分类模型
device: str = "cpu"                                 # → "cuda" 切 GPU
```

如果是换更大模型（如 `bert-base-chinese` → `hfl/chinese-roberta-wwm-ext-large`），同步调整 `max_length: 128`。

### 更换向量模型

两处：

**SQL Agent** (`sql_agent/clients/embedding_client_manager.py`)：
```python
self.client = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",  # → 换成 "BAAI/bge-m3" 或其他
)
```

**RAG Agent** (`rag_agent/config.py`)：
```python
embedding_model_name: str = "BAAI/bge-m3"        # → 换成其他
reranker_model_name: str = "BAAI/bge-reranker-v2-m3"  # → 换成其他
vector_size: int = 1024                           # 同步改为新模型的维度
```

换后重建知识库：`python -m rag_agent.scripts.build_kb --rebuild`

### 更换向量数据库

Qdrant 客户端在以下位置初始化：
- SQL Agent: `sql_agent/clients/qdrant_client_manager.py`
- RAG Agent: `rag_agent/qdrant_store.py`

换成 Milvus/Weaviate 等，需改对应的初始化代码和搜索/写入逻辑。Qdrant 的 HTTP 地址在 `conf/app_config.yaml`：

```yaml
qdrant:
  host: localhost    # → 生产环境 IP
  port: 6333
```

### 更换搜索引擎

网络搜索在 `tools/deep_search_tools.py` 和 `tools/tavily_tools.py`：

```python
from tavily import TavilyClient                               # → 换成其他 SDK
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
```

换成 SerpAPI / Brave Search / Bing API 等，改客户端初始化 + 调用方法。并行搜索层不用动。

### 更换关系数据库

在 `conf/app_config.yaml`：

```yaml
db_meta:
  host: localhost       # → 生产环境 IP
  port: 3306
  user: root
  password: your_pass
  database: meta

db_dw:
  host: localhost
  database: pharma_db    # → 业务库名
```

换成 PostgreSQL：MySQL 特有的 `asyncmy` 驱动和少量 SQL 语法差异需微调 `sql_agent/` 下的 Repository 层。

### 更换重排序模型

`rag_agent/config.py`：
```python
reranker_model_name: str = "BAAI/bge-reranker-v2-m3"  # → 换成其他 Cross-Encoder
```

动态 TopK 阈值同步可调：
```python
rerank_gap_ratio: float = 0.25   # 相对断崖阈值（调大=更宽容，保留更多结果）
rerank_gap_abs: float = 0.5      # 绝对断崖阈值
```

## License

MIT
