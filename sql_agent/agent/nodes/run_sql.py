from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger


async def run_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('运行SQL语句中')

    sql = state['sql']
    dw_mysql_repository = runtime.context['dw_mysql_repository']

    result = await dw_mysql_repository.run_sql(sql)
    logger.info(f"生成的SQL: {sql}")
    logger.info(f"执行的结果: {result}")

    # 透传 SQL 和执行结果，方便评测时捕获
    output = {
        "sql": sql,
        "result": str(result),
    }
    return {
        'messages': [{'role': 'assistant', 'content': str(output)}],
        'sql': sql,  # 保证 state 中 sql 字段始终是最新值
    }
