"""
数据库查询助手 (Database Query Agent)

基于 Agent_sql 项目架构升级：从简单的 3-tool 模式升级为 LangGraph 流水线模式。

流水线流程:
  提取关键词 → 向量检索(列/指标/值,3路并行) → 合并检索结果
  → LLM过滤表/指标 → 补充上下文 → 生成SQL → EXPLAIN校验
  → (通过则执行, 失败则LLM纠错后执行)

技术栈:
  - LangGraph StateGraph: 多节点流水线编排
  - Qdrant: 列/指标的向量语义检索
  - Elasticsearch: 字段取值的全文检索
  - HuggingFace BgeEmbeddings: 中文语义向量化
  - SQLAlchemy async: MySQL 异步连接 (meta库 + dw库)
  - jieba: 中文关键词提取

依赖说明:
  pip install loguru jieba qdrant-client elasticsearch sqlalchemy[asyncio]
       asyncmy omegaconf langchain-community sentence-transformers

前置条件 (运行时):
  1. 运行 python -m sql_agent.scripts.build_meta_knowledge -c conf/meta_config.yaml
     将 pharma_db 的元数据导入 meta 库 + Qdrant + ES
  2. 确保 Qdrant(6333)、Elasticsearch(9200)、MySQL(3306) 服务运行中
"""

from deepagents import CompiledSubAgent
from agent.prompts import sub_agents_config

from sql_agent.agent.graph import sql_graph
from sql_agent.agent.state import DataAgentState
from sql_agent.agent.context import DataAgentContext

from sql_agent.clients.qdrant_client_manager import qdrant_client_manager
from sql_agent.clients.embedding_client_manager import embedding_client_manager
from sql_agent.clients.es_client_manager import es_client_manager
from sql_agent.clients.mysql_client_manager import meta_sql_client_manager, dw_sql_client_manager

from sql_agent.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from sql_agent.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from sql_agent.repositories.es.value_es_repository import ValueEsRepository
from sql_agent.repositories.mysql.meta.meta_mysql_repository import MetaMysqlRepository
from sql_agent.repositories.mysql.dw.Dw_Mysql_repository import DwMysqlRepository


def _init_infrastructure():
    """初始化所有基础设施客户端"""
    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    meta_sql_client_manager.init()
    dw_sql_client_manager.init()


def _build_context(meta_session, dw_session) -> DataAgentContext:
    """构建 LangGraph 流水线所需的运行时上下文"""
    return DataAgentContext(
        column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
        embedding_client=embedding_client_manager.client,
        metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
        value_es_repository=ValueEsRepository(es_client_manager.client),
        meta_mysql_repository=MetaMysqlRepository(meta_session),
        dw_mysql_repository=DwMysqlRepository(dw_session),
    )


# CompiledSubAgent 封装：将 LangGraph 流水线包装为 DeepAgents 可调度的子代理
# 主代理通过 task(subagent_type="数据库查询助手", description="查询xxx") 调用
# 子代理内部运行完整的 提取→检索→过滤→生成→校验→执行 流水线
database_query_agent = CompiledSubAgent(
    name=sub_agents_config["db"].get("name", "数据库查询助手"),
    description=sub_agents_config["db"].get("description", ""),
    runnable=sql_graph,  # 编译后的 LangGraph StateGraph
)

# 注意：运行时需要先调用 _init_infrastructure() 初始化所有客户端，
# 然后通过 _build_context() 构建 context 传入 graph.astream(state, context=context)
# 实际集成方式参见 agent/main_agent.py 中的 run_deep_agent
