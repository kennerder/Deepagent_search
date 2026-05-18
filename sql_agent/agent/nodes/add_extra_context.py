from datetime import date
from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.state import DataAgentState, DateInfoState, DBInfoState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.repositories.mysql.dw.Dw_Mysql_repository import DwMysqlRepository


async def add_extra_context(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('添加上下文信息')

    dw_mysql_repository: DwMysqlRepository = runtime.context["dw_mysql_repository"]

    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    weekday = today.strftime("%A")
    quarter = f"Q{(today.month - 1) // 3 + 1}"
    date_info = DateInfoState(date=date_str, weekday=weekday, quarter=quarter)

    db = await dw_mysql_repository.get_db_info()
    db_info = DBInfoState(**db)
    logger.info(f"数据库信息{db_info}")
    logger.info(f"日期信息{date_info}")
    return {'date_info': date_info, 'db_info': db_info}
