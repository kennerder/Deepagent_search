from typing import TypedDict
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from sql_agent.repositories.es.value_es_repository import ValueEsRepository
from sql_agent.repositories.mysql.dw.Dw_Mysql_repository import DwMysqlRepository
from sql_agent.repositories.mysql.meta.meta_mysql_repository import MetaMysqlRepository
from sql_agent.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from sql_agent.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


class DataAgentContext(TypedDict):
    column_qdrant_repository: ColumnQdrantRepository
    embedding_client: HuggingFaceBgeEmbeddings
    metric_qdrant_repository: MetricQdrantRepository
    value_es_repository: ValueEsRepository
    meta_mysql_repository: MetaMysqlRepository
    dw_mysql_repository: DwMysqlRepository
