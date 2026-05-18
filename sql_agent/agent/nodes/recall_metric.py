from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.llm import llm
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.entities.metric_info import MetricInfo
from sql_agent.prompt.prompt_loader import load_prompt


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('正在召回指标信息')

    keywords = state['keywords']
    query = state['query']
    embedding_client = runtime.context['embedding_client']
    metric_qdrant_repository = runtime.context['metric_qdrant_repository']

    prompt = PromptTemplate(template=load_prompt('extend_keywords_for_metric_recall'),
                            input_variables=['query'])
    chain = prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({'query': query})
    keywords = set(keywords + result)
    logger.info(f'扩展之后的指标关键词列表为{keywords}')

    metric_info_map: dict[str, MetricInfo] = {}
    for keyword in keywords:
        embedding = await embedding_client.aembed_query(keyword)
        current_metric_infos: list[MetricInfo] = await metric_qdrant_repository.search(
            embedding,
            score_threshold=0.5,
            limit=3,
        )
        for current_metric_info in current_metric_infos:
            if current_metric_info.id not in metric_info_map:
                metric_info_map[current_metric_info.id] = current_metric_info
    retrieved_metric_infos: list[MetricInfo] = list(metric_info_map.values())
    logger.info(f'检索到的指标信息为{list(metric_info_map.keys())}')
    return {'retrieved_metric_infos': retrieved_metric_infos}
