import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.llm import llm
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.prompt.prompt_loader import load_prompt


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('更正sql语句')

    table_infos = state['table_infos']
    metric_infos = state['metric_infos']
    date_info = state['date_info']
    db_info = state['db_info']
    query = state['query']
    sql = state['sql']
    error = state['error']

    prompt = PromptTemplate(template=load_prompt('correct_sql'),
                            input_variables=['query', 'metric_infos', 'table_infos',
                                             'date_info', 'db_info', 'sql', 'error'])

    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke({'query': query,
                                  'metric_infos': yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
                                  'table_infos': yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
                                  'date_info': yaml.dump(date_info, allow_unicode=True, sort_keys=False),
                                  'db_info': yaml.dump(db_info, allow_unicode=True, sort_keys=False),
                                  'sql': yaml.dump(sql, allow_unicode=True, sort_keys=False),
                                  'error': yaml.dump(error, allow_unicode=True, sort_keys=False)})

    logger.info(f"校正后的SQL为: {result}")
    return {'sql': result}
