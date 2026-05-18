from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sql_agent.entities.column_info import ColumnInfo
from sql_agent.entities.column_metric import ColumnMetric
from sql_agent.entities.metric_info import MetricInfo
from sql_agent.entities.table_info import TableInfo
from sql_agent.models.column_info import ColumnInfoMySQL
from sql_agent.models.table_info import TableInfoMySQL
from sql_agent.repositories.mysql.meta.mapper.column_info_mapper import ColumnInfoMapper
from sql_agent.repositories.mysql.meta.mapper.column_metric_mapper import ColumnMetricMapper
from sql_agent.repositories.mysql.meta.mapper.metric_info_mapper import MetricInfoMapper
from sql_agent.repositories.mysql.meta.mapper.table_info_mapper import TableInfoMapper


class MetaMysqlRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    def save_table_infos(self, table_infos: list[TableInfo]):
        self.session.add_all([TableInfoMapper.to_model(table_info) for table_info in table_infos])

    def save_column_infos(self, column_infos: list[ColumnInfo]):
        self.session.add_all([ColumnInfoMapper.to_model(column_info) for column_info in column_infos])

    async def save_metric_infos(self, metric_infos: list[MetricInfo]):
        self.session.add_all([MetricInfoMapper.to_model(metric_info) for metric_info in metric_infos])

    async def save_column_metrics(self, column_metrics: list[ColumnMetric]):
        self.session.add_all([ColumnMetricMapper.to_model(column_metric) for column_metric in column_metrics])

    async def get_column_info_by_id(self, id) -> ColumnInfo | None:
        column_info: ColumnInfoMySQL | None = await self.session.get(
            ColumnInfoMySQL,
            id
        )
        if column_info:
            return ColumnInfoMapper.to_entity(column_info)
        else:
            return None

    async def get_table_info_by_id(self, id: str) -> TableInfo | None:
        table_info: TableInfoMySQL | None = await self.session.get(
            TableInfoMySQL,
            id
        )
        if table_info:
            return TableInfoMapper.to_entity(table_info)
        else:
            return None

    async def get_key_columns_by_table_id(self, table_id) -> list[ColumnInfo]:
        sql = ("select * from column_info where table_id "
               "= :table_id and role in ('primary_key','foreign_key')")
        result = await self.session.execute(text(sql), {"table_id": table_id})
        return [ColumnInfo(**dict(row)) for row in result.mappings().fetchall()]
