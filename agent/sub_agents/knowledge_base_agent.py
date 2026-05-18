"""
RAG 知识库助手 (Knowledge Base Agent)

从 RAGFlow 远程服务 替换为 本地 RAG 检索管线。

检索管线:
  用户问题 → HyDE假设文档生成 → 查询增强 → BGE-M3向量化
  → Qdrant子块检索 → 父块追溯 → BGE Reranker v2-m3 重排序
  → 返回 TopK 文档

技术栈:
  - 父子文档切分 (Markdown标题优先, 子块512/父块2048)
  - HyDE (Hypothetical Document Embedding) 问题假设增强
  - BGE-M3 向量化 + Qdrant 父子双集合存储
  - BGE Reranker v2-m3 Cross-Encoder 精排 + 动态TopK
  - MD5 去重 + 增量构建

工具:
  - search_knowledge_base: 检索知识库
  - get_knowledge_base_status: 查看知识库状态
"""

from typing import Annotated, Optional
from langchain_core.tools import tool

from agent.prompts import sub_agents_config
from api.monitor import monitor
from rag_agent.retriever import get_retriever
from rag_agent.builder import KnowledgeBaseBuilder


@tool
def search_knowledge_base(
    query: Annotated[str, "要在知识库中检索的问题或关键词"],
    top_k: Annotated[int, "返回的文档数量，默认5"] = 5,
) -> str:
    """
    【核心工具】在本地知识库中检索相关文档。

    内部流程:
    1. HyDE: LLM 生成假设文档增强查询语义
    2. 向量检索: BGE-M3 → Qdrant 子块检索
    3. 父块追溯: 命中子块 → 返回完整父块上下文
    4. 重排序: BGE Reranker v2-m3 精细打分
    5. 动态TopK: 检测分数断崖自动截断

    适用场景:
    - 需要查找企业内部文档、规章制度、技术手册等
    - 需要基于私有知识回答问题时

    Args:
        query: 检索问题
        top_k: 返回文档数

    Returns:
        格式化的检索结果文本
    """
    monitor.report_tool("知识库检索工具", {"query": query, "top_k": top_k})

    try:
        retriever = get_retriever()
        # 同步检索（工具调用是同步的）
        results = retriever.retrieve(query, top_k=top_k)

        if not results:
            return "未在知识库中找到相关内容。"

        # 格式化为可读文本
        formatted = retriever.format_context(results)

        monitor.report_tool("知识库检索结果", {
            "hit_count": len(results),
            "top_score": results[0].get("rerank_score", 0) if results else 0,
        })

        return f"知识库检索结果（共 {len(results)} 条）:\n\n{formatted}"

    except Exception as e:
        return f"知识库检索失败: {str(e)}"


@tool
def get_knowledge_base_status(
    dummy: Annotated[str, "无需传参，直接调用"] = "",
) -> str:
    """
    查看本地知识库的状态信息。

    返回知识库中的文档数量、块数量、构建时间等统计信息。
    当用户询问"知识库有什么内容"或"知识库状态"时调用。
    """
    monitor.report_tool("知识库状态查询")

    try:
        builder = KnowledgeBaseBuilder()
        stats = builder.get_stats()

        return (
            f"知识库状态:\n"
            f"  - 已处理文件数: {stats['processed_files']}\n"
            f"  - 父块数量: {stats['parent_chunks']}\n"
            f"  - 子块数量: {stats['child_chunks']}\n"
            f"  - 文档目录: {stats['documents_dir']}\n"
        )
    except Exception as e:
        return f"查询知识库状态失败: {str(e)}"


# DeepAgents 子代理配置
knowledge_base_agent = {
    "name": sub_agents_config["ragflow"].get("name", "RAGFlow助手"),
    "description": sub_agents_config["ragflow"].get("description", ""),
    "system_prompt": sub_agents_config["ragflow"].get("system_prompt", ""),
    "tools": [search_knowledge_base, get_knowledge_base_status]
}
