from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.state import DataAgentState, TableInfoState, ColumnInfoState, MetricInfoState
from langgraph.runtime import Runtime
from sql_agent.entities.column_info import ColumnInfo
from sql_agent.entities.metric_info import MetricInfo
from sql_agent.entities.table_info import TableInfo
from sql_agent.entities.value_info import ValueInfo
from sql_agent.repositories.mysql.meta.meta_mysql_repository import MetaMysqlRepository


async def merge_retrieved_info(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('合并检索信息')

    retrieved_column_infos: list[ColumnInfo] = state["retrieved_column_infos"]
    retrieved_metric_infos: list[MetricInfo] = state["retrieved_metric_infos"]
    retrieved_value_infos: list[ValueInfo] = state["retrieved_value_infos"]

    meta_mysql_repository: MetaMysqlRepository = runtime.context["meta_mysql_repository"]

    # 将指标信息的相关字段信息添加到字段信息中
    retrieved_column_infos_map: dict[str, ColumnInfo] = {
        retrieved_column_info.id: retrieved_column_info for
        retrieved_column_info in retrieved_column_infos
    }
    for retrieved_metric_info in retrieved_metric_infos:
        for relevant_column in retrieved_metric_info.relevant_columns:
            if relevant_column not in retrieved_column_infos_map:
                column_info: ColumnInfo = \
                    await meta_mysql_repository.get_column_info_by_id(relevant_column)
                retrieved_column_infos_map[relevant_column] = column_info

    # 将字段取值添加到其所述字段的examples中
    for retrieved_value_info in retrieved_value_infos:
        value = retrieved_value_info.value
        column_id = retrieved_value_info.column_id
        if column_id not in retrieved_column_infos_map:
            column_info: ColumnInfo = await meta_mysql_repository.get_column_info_by_id(column_id)
            retrieved_column_infos_map[column_id] = column_info
        if value not in retrieved_column_infos_map[column_id].examples:
            retrieved_column_infos_map[column_id].examples.append(value)

    # 按照表对字段信息进行整理分组
    table_to_columns_map: dict[str, list[ColumnInfo]] = {}
    for column_info in retrieved_column_infos_map.values():
        table_id = column_info.table_id
        if table_id not in table_to_columns_map:
            table_to_columns_map[table_id] = []
        table_to_columns_map[table_id].append(column_info)

    # 强制为每个表添加上主外键
    for table_id in table_to_columns_map.keys():
        key_columns: list[ColumnInfo] \
            = await meta_mysql_repository.get_key_columns_by_table_id(table_id)
        column_ids = [column_info.id for column_info in table_to_columns_map[table_id]]
        for key_column in key_columns:
            if key_column.id not in column_ids:
                table_to_columns_map[table_id].append(key_column)

    table_infos: list[TableInfoState] = []
    for table_id, column_infos in table_to_columns_map.items():
        table_info: TableInfo = await meta_mysql_repository.get_table_info_by_id(table_id)
        columns = [ColumnInfoState(
            name=column_info.name,
            type=column_info.type,
            role=column_info.role,
            example=column_info.examples,
            description=column_info.description,
            alias=column_info.alias,
        ) for column_info in column_infos]
        table_info_state = TableInfoState(
            name=table_info.name,
            role=table_info.role,
            columns=columns,
            description=table_info.description,
        )
        table_infos.append(table_info_state)

    # 处理指标信息
    metric_infos: list[MetricInfoState] = [MetricInfoState(
        name=retrieved_metric_info.name,
        description=retrieved_metric_info.description,
        relevant_columns=retrieved_metric_info.relevant_columns,
        alias=retrieved_metric_info.alias,
    ) for retrieved_metric_info in retrieved_metric_infos]

    return {
        "table_infos": table_infos,
        "metric_infos": metric_infos,
    }
