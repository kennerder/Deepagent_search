import asyncio

from elasticsearch import AsyncElasticsearch
from sql_agent.conf.app_config import ESConfig, app_config


class EsClientManager:
    def __init__(self, config: ESConfig):
        self.client: AsyncElasticsearch | None = None
        self.config: ESConfig = config

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = AsyncElasticsearch(hosts=[self._get_url()])

    async def close(self):
        await self.client.close()


es_client_manager = EsClientManager(app_config.es)
