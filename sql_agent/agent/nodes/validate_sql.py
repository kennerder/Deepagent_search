from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.repositories.mysql.dw.Dw_Mysql_repository import DwMysqlRepository


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('正在验证SQL语句')

    sql = state['sql']
    dw_mysql_repository: DwMysqlRepository = runtime.context['dw_mysql_repository']
    try:
        await dw_mysql_repository.validate_sql(sql)
        logger.info(f"SQL语法正确")
        return {'error': None}
    except Exception as e:
        logger.info(f"SQL语法错误: {str(e)}")
        return {'error': str(e)}
