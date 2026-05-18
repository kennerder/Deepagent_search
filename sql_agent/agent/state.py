from typing import TypedDict

from sql_agent.entities.column_info import ColumnInfo
from sql_agent.entities.metric_info import MetricInfo
from sql_agent.entities.value_info import ValueInfo


class MetricInfoState(TypedDict):
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]


class ColumnInfoState(TypedDict):
    name: str
    type: str
    role: str
    example: list
    description: str
    alias: list[str]


class TableInfoState(TypedDict):
    name: str
    role: str
    description: str
    columns: list[ColumnInfoState]


class DateInfoState(TypedDict):
    date: str
    weekday: str
    quarter: str


class DBInfoState(TypedDict):
    dialect: str
    version: str


class DataAgentState(TypedDict):
    query: str
    keywords: list[str]
    retrieved_column_infos: list[ColumnInfo]
    retrieved_metric_infos: list[MetricInfo]
    retrieved_value_infos: list[ValueInfo]
    error: str

    table_infos: list[TableInfoState]
    metric_infos: list[MetricInfoState]

    sql: str
    date_info: DateInfoState
    db_info: DBInfoState
