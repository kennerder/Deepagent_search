from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from sql_agent.agent.llm import llm
from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime

from sql_agent.entities.column_info import ColumnInfo
from sql_agent.prompt.prompt_loader import load_prompt
from sql_agent.core.log import logger


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('正在召回字段信息')
    keywords = state['keywords']
    query = state['query']
    column_qdrant_repository = runtime.context['column_qdrant_repository']
    embedding_client = runtime.context['embedding_client']

    prompt = PromptTemplate(template=load_prompt('extend_keywords_for_column_recall'),
                            input_variables=['query'])
    chain = prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({'query': query})
    keywords = set(keywords + result)
    logger.info(f'扩展之后的字段关键词列表为{keywords}')

    column_info_map: dict[str, ColumnInfo] = {}
    for keyword in keywords:
        embedding = await embedding_client.aembed_query(keyword)
        current_column_infos: list[ColumnInfo] = await column_qdrant_repository.search(
            embedding,
            score_threshold=0.5,
            limit=3,
        )
        for current_column_info in current_column_infos:
            if current_column_info.id not in column_info_map:
                column_info_map[current_column_info.id] = current_column_info
    retrieved_column_infos: list[ColumnInfo] = list(column_info_map.values())
    logger.info(f'检索到的字段信息为{list(column_info_map.keys())}')
    return {'retrieved_column_infos': retrieved_column_infos}
