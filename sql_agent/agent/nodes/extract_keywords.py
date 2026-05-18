import jieba.analyse

from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.state import DataAgentState
from langgraph.runtime import Runtime
from sql_agent.core.log import logger


async def extract_keywords(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer('提取关键词')
    query = state['query']

    allow_pos = (
        "n", "nr", "ns", "nt", "nz",
        "v", "vn",
        "a", "an",
        "eng", "i", "l",
    )

    keywords = jieba.analyse.extract_tags(query, allowPOS=allow_pos)
    keywords = list(set(keywords + [query]))
    logger.info(f"抽取的关键词为{keywords}")
    return {"keywords": keywords}
