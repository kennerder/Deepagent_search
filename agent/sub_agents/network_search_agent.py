"""
网络搜索助手 (Network Search Agent)

搜索策略:
  多角度深度搜索 — LLM 改写 Query → 硬编码 3~5 轮并行搜索 → URL 去重聚合
  通过多角度、多关键词的广域搜索，构建高密度全局上下文，有效降低单一信息源偏见。

技术栈:
  - Tavily Search API: 网络搜索引擎
  - LLM Query Rewriting: 多角度搜索词生成
  - ThreadPoolExecutor: 并行搜索执行
  - URL 去重 + 高密度上下文聚合
"""

from agent.prompts import sub_agents_config
from tools.deep_search_tools import deep_search

network_search_agent = {
    "name": sub_agents_config["tavily"].get("name", ""),
    "description": sub_agents_config["tavily"].get("description", ""),
    "system_prompt": sub_agents_config["tavily"].get("system_prompt", ""),
    "tools": [deep_search],
}
