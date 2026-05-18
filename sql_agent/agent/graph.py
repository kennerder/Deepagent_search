import os

# 必须放在导入 HuggingFaceBgeEmbeddings 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from sql_agent.agent.context import DataAgentContext
from sql_agent.agent.nodes.add_extra_context import add_extra_context
from sql_agent.agent.nodes.correct_sql import correct_sql
from sql_agent.agent.nodes.extract_keywords import extract_keywords
from sql_agent.agent.nodes.filter_metric import filter_metric
from sql_agent.agent.nodes.filter_table import filter_table
from sql_agent.agent.nodes.generate_sql import generate_sql
from sql_agent.agent.nodes.merge_retrieved_info import merge_retrieved_info
from sql_agent.agent.nodes.recall_column import recall_column
from sql_agent.agent.nodes.recall_metric import recall_metric
from sql_agent.agent.nodes.recall_value import recall_value
from sql_agent.agent.nodes.run_sql import run_sql
from sql_agent.agent.nodes.validate_sql import validate_sql
from sql_agent.agent.state import DataAgentState

graph_builder = StateGraph(state_schema=DataAgentState, context_schema=DataAgentContext)

# 添加节点
graph_builder.add_node("extract_keywords", extract_keywords)
graph_builder.add_node("recall_column", recall_column)
graph_builder.add_node("recall_value", recall_value)
graph_builder.add_node("recall_metric", recall_metric)
graph_builder.add_node("merge_retrieved_info", merge_retrieved_info)
graph_builder.add_node("filter_metric", filter_metric)
graph_builder.add_node("filter_table", filter_table)
graph_builder.add_node("add_extra_context", add_extra_context)
graph_builder.add_node("generate_sql", generate_sql)
graph_builder.add_node("validate_sql", validate_sql)
graph_builder.add_node("correct_sql", correct_sql)
graph_builder.add_node("run_sql", run_sql)

# 添加关系
graph_builder.add_edge(START, "extract_keywords")
graph_builder.add_edge("extract_keywords", "recall_column")
graph_builder.add_edge("extract_keywords", "recall_value")
graph_builder.add_edge("extract_keywords", "recall_metric")
graph_builder.add_edge("recall_column", "merge_retrieved_info")
graph_builder.add_edge("recall_value", "merge_retrieved_info")
graph_builder.add_edge("recall_metric", "merge_retrieved_info")
graph_builder.add_edge("merge_retrieved_info", "filter_table")
graph_builder.add_edge("merge_retrieved_info", "filter_metric")
graph_builder.add_edge("filter_table", "add_extra_context")
graph_builder.add_edge("filter_metric", "add_extra_context")
graph_builder.add_edge("add_extra_context", "generate_sql")
graph_builder.add_edge("generate_sql", "validate_sql")

graph_builder.add_conditional_edges(
    source='validate_sql',
    path=lambda state: "run_sql" if state['error'] is None else "correct_sql",
    path_map={"run_sql": "run_sql", "correct_sql": "correct_sql"},
)
graph_builder.add_edge("correct_sql", "run_sql")
graph_builder.add_edge("run_sql", END)

sql_graph = graph_builder.compile()
