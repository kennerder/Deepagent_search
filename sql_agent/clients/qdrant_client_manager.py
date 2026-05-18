import asyncio

from qdrant_client import AsyncQdrantClient

from sql_agent.conf.app_config import QdrantConfig, app_config


class QdrantClientManager:
    def __init__(self, config: QdrantConfig):
        self.client: AsyncQdrantClient | None = None
        self.config: QdrantConfig = config

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = AsyncQdrantClient(url=self._get_url())

    async def close(self):
        await self.client.close()


qdrant_client_manager = QdrantClientManager(app_config.qdrant)
