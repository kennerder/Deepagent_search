import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.llm import llm
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger
from sql_agent.prompt.prompt_loader import load_prompt


async def filter_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('过滤多余指标信息')

    metric_infos = state['metric_infos']
    query = state['query']

    prompt = PromptTemplate(template=load_prompt('filter_metric_info'),
                            input_variables=['query', 'metric_infos'])
    chain = prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({'query': query,
                                  'metric_infos': yaml.dump(metric_infos, allow_unicode=True, sort_keys=False)})

    filtered_metric_infos = [metric_info for metric_info in metric_infos if
                             metric_info["name"] in result]

    logger.info(f"过滤后的指标信息：{[m['name'] for m in filtered_metric_infos]}")
    return {"metric_infos": filtered_metric_infos}
