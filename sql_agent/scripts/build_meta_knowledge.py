import sys
from pathlib import Path
import argparse
import asyncio

# 确保项目根目录在 sys.path 中
base_dir = str(Path(__file__).resolve().parents[2])
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from sql_agent.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from sql_agent.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from sql_agent.clients.es_client_manager import es_client_manager
from sql_agent.repositories.es.value_es_repository import ValueEsRepository
from sql_agent.clients.embedding_client_manager import embedding_client_manager
from sql_agent.clients.qdrant_client_manager import qdrant_client_manager
from sql_agent.clients.mysql_client_manager import meta_sql_client_manager, dw_sql_client_manager
from sql_agent.repositories.mysql.dw.Dw_Mysql_repository import DwMysqlRepository
from sql_agent.repositories.mysql.meta.meta_mysql_repository import MetaMysqlRepository
from sql_agent.services.meta_knowledge_service import MetaKnowledgeService


async def build(config_path: Path):
    meta_sql_client_manager.init()
    dw_sql_client_manager.init()
    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    async with (meta_sql_client_manager.session_factory() as meta_session,
                dw_sql_client_manager.session_factory() as dw_session):
        column_qdrant_repository = ColumnQdrantRepository(qdrant_client_manager.client)
        metric_qdrant_repository = MetricQdrantRepository(qdrant_client_manager.client)
        value_es_repository = ValueEsRepository(es_client_manager.client)
        meta_mysql_repository = MetaMysqlRepository(meta_session)
        dw_mysql_repository = DwMysqlRepository(dw_session)
        meta_knowledge_service = MetaKnowledgeService(meta_mysql_repository=meta_mysql_repository,
                                                      dw_mysql_repository=dw_mysql_repository,
                                                      column_qdrant_repository=column_qdrant_repository,
                                                      embedding_client=embedding_client_manager.client,
                                                      value_es_repository=value_es_repository,
                                                      metric_qdrant_repository=metric_qdrant_repository)
        await meta_knowledge_service.build(config_path)
    await meta_sql_client_manager.close()
    await dw_sql_client_manager.close()
    await qdrant_client_manager.close()
    await es_client_manager.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf')
    args = parser.parse_args()
    config_path = args.conf
    asyncio.run(build(config_path))
