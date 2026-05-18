import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

from sql_agent.conf.app_config import DBConfig, app_config


class MySqlClientManager:
    def __init__(self, config: DBConfig):
        self.engine: AsyncEngine | None = None
        self.config: DBConfig = config
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

    def _get_url(self):
        return (f"mysql+asyncmy://{self.config.user}:{self.config.password}"
                f"@{self.config.host}/{self.config.database}?charset=utf8mb4")

    def init(self):
        self.engine = create_async_engine(url=self._get_url())
        self.session_factory = async_sessionmaker(self.engine, autoflush=True, expire_on_commit=False)

    async def close(self):
        await self.engine.dispose()


meta_sql_client_manager = MySqlClientManager(app_config.db_meta)
dw_sql_client_manager = MySqlClientManager(app_config.db_dw)
