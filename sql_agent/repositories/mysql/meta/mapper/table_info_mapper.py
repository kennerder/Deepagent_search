from dataclasses import asdict

from sql_agent.entities.table_info import TableInfo
from sql_agent.models.table_info import TableInfoMySQL


class TableInfoMapper:
    @staticmethod
    def to_entity(model: TableInfoMySQL) -> TableInfo:
        return TableInfo(
            id=model.id,
            name=model.name,
            role=model.role,
            description=model.description,
        )

    @staticmethod
    def to_model(entity: TableInfo) -> TableInfoMySQL:
        return TableInfoMySQL(**asdict(entity))
