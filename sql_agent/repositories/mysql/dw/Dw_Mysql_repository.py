from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class DwMysqlRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_column_types(self, table_name):
        sql = f"show columns from {table_name}"
        result = await self.session.execute(text(sql))
        result_dict = result.mappings().fetchall()
        return {row['Field']: row['Type'] for row in result_dict}

    async def get_column_values(self, table_name, column_name, limit=10):
        sql = f"select distinct {column_name} from {table_name} limit {limit}"
        result = await self.session.execute(text(sql))
        return [row[0] for row in result.fetchall()]

    async def get_db_info(self):
        sql = 'select version()'
        result = await self.session.execute(text(sql))
        version = result.scalar()

        dialect = 'mysql'
        return {'dialect': dialect, 'version': version}

    async def validate_sql(self, sql: str):
        sql = f'explain {sql}'
        await self.session.execute(text(sql))

    async def run_sql(self, sql):
        result = await self.session.execute(text(sql))
        return [dict(row) for row in result.mappings().fetchall()]
