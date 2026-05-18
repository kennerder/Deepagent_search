from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.llm import llm
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.entities.value_info import ValueInfo
from sql_agent.prompt.prompt_loader import load_prompt


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('正在召回取值信息')

    keywords = state['keywords']
    query = state['query']
    value_es_repository = runtime.context['value_es_repository']

    prompt = PromptTemplate(template=load_prompt('extend_keywords_for_value_recall'),
                            input_variables=['query'])
    chain = prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({'query': query})
    keywords = set(keywords + result)
    logger.info(f'扩展之后的取值关键词列表为{keywords}')

    value_infos_map: dict[str, ValueInfo] = {}
    for keyword in keywords:
        current_value_infos: list[ValueInfo] = await value_es_repository.search(keyword)
        for current_value_info in current_value_infos:
            if current_value_info.id not in value_infos_map:
                value_infos_map[current_value_info.id] = current_value_info
    retrieved_value_infos: list[ValueInfo] = list(value_infos_map.values())
    logger.info(f'检索到的字段取值：{list(value_infos_map.keys())}')
    return {'retrieved_value_infos': retrieved_value_infos}
