import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.llm import llm
from sql_agent.agent.state import DataAgentState, TableInfoState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.prompt.prompt_loader import load_prompt


async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('过滤多余表信息')

    table_infos = state['table_infos']
    query = state['query']

    prompt = PromptTemplate(template=load_prompt('filter_table_info'),
                            input_variables=['query', 'table_infos'])
    chain = prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({'query': query,
                                  'table_infos': yaml.dump(table_infos, allow_unicode=True, sort_keys=False)})

    filtered_table_infos: list[TableInfoState] = []
    for table_info in table_infos:
        if table_info["name"] in result:
            table_info["columns"] = [column_info for column_info in table_info["columns"] if
                                     column_info["name"] in result[table_info["name"]]]
            filtered_table_infos.append(table_info)

    logger.info(f"过滤后的表信息：{[t['name'] for t in filtered_table_infos]}")
    return {"table_infos": filtered_table_infos}
