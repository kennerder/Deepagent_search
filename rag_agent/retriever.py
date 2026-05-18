"""
检索模块：完整的 RAG 检索管线

检索流程:
  1. HyDE 生成: 用户查询 → LLM 生成假设文档
  2. 查询增强: 原查询 + HyDE 文档 → 富语义文本
  3. 向量检索: 增强文本 → embedding → Qdrant 子块检索
  4. 父块追溯: 子块命中 → 查 parent_id → 返回父块完整内容
  5. 重排序: BGE Reranker 精排父块
  6. 结果返回: 动态 TopK 截断后的文档列表
"""

from typing import List, Dict, Optional

from rag_agent.config import rag_config
from rag_agent.embedder import get_embedder
from rag_agent.qdrant_store import get_qdrant_store
from rag_agent.hyde import get_hyde_generator
from rag_agent.reranker import get_reranker


class RAGRetriever:
    """
    RAG 检索器：串联 HyDE → Embedding → Qdrant → Rerank 全流程。

    使用方式:
        retriever = RAGRetriever()
        results = await retriever.aretrieve("如何设置烫金参数？")
        # → [{"content": "...", "title": "...", "rerank_score": 0.95}, ...]
    """

    def __init__(self):
        self.embedder = get_embedder()
        self.qdrant = get_qdrant_store()
        self.hyde = get_hyde_generator()
        self.reranker = get_reranker()

    async def aretrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        异步检索：完整管线。

        Args:
            query: 用户查询
            top_k: 最终返回数量

        Returns:
            重排后的文档列表
        """
        # Step 1: HyDE 生成假设文档
        hyde_doc = await self.hyde.agenerate(query)

        # Step 2: 查询增强
        enriched_query = self.hyde.enrich_query(query, hyde_doc)

        # Step 3: 向量化 + Qdrant 检索子块
        query_vector = self.embedder.encode(enriched_query)[0]
        child_results = self.qdrant.search(
            query_vector,
            top_k=rag_config.retrieval_top_k,
        )

        if not child_results:
            return []

        # Step 4: 追溯到父块
        parent_docs = self.qdrant.get_parents_by_child_ids(child_results)

        if not parent_docs:
            # 如果没有父块，直接用子块内容
            parent_docs = [
                {
                    "id": r["id"],
                    "content": r.get("payload", {}).get("content", ""),
                    "title": r.get("payload", {}).get("title", ""),
                    "file_name": r.get("payload", {}).get("file_name", ""),
                    "child_score": r.get("score"),
                }
                for r in child_results
            ]

        # Step 5: Rerank 重排序
        reranked = self.reranker.rerank(
            query=query,  # 用原始查询重排（不用 HyDE）
            documents=parent_docs,
            top_k=top_k,
        )

        return reranked

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        同步检索：完整管线。

        Args:
            query: 用户查询
            top_k: 最终返回数量

        Returns:
            重排后的文档列表
        """
        # Step 1: HyDE
        hyde_doc = self.hyde.generate(query)

        # Step 2: 查询增强
        enriched_query = self.hyde.enrich_query(query, hyde_doc)

        # Step 3: 向量检索子块
        query_vector = self.embedder.encode(enriched_query)[0]
        child_results = self.qdrant.search(query_vector)

        if not child_results:
            return []

        # Step 4: 追溯父块
        parent_docs = self.qdrant.get_parents_by_child_ids(child_results)

        if not parent_docs:
            parent_docs = [
                {
                    "id": r["id"],
                    "content": r.get("payload", {}).get("content", ""),
                    "title": r.get("payload", {}).get("title", ""),
                    "file_name": r.get("payload", {}).get("file_name", ""),
                    "child_score": r.get("score"),
                }
                for r in child_results
            ]

        # Step 5: Rerank
        reranked = self.reranker.rerank(
            query=query,
            documents=parent_docs,
            top_k=top_k,
        )

        return reranked

    def format_context(self, documents: List[Dict], max_length: int = 4000) -> str:
        """
        将检索结果格式化为 LLM 可用的上下文字符串。

        Args:
            documents: 检索结果列表
            max_length: 最大总长度

        Returns:
            格式化的上文字符串
        """
        parts = []
        total_len = 0

        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("file_name", "未知来源")
            score = doc.get("rerank_score", doc.get("child_score", 0))

            header = f"[文档{i}] 来源: {source} (相关度: {score:.2f})"
            chunk = f"{header}\n{content}"

            if total_len + len(chunk) > max_length:
                # 截断最后一个文档
                remaining = max_length - total_len - 100
                if remaining > 0:
                    parts.append(f"{header}\n{content[:remaining]}...")
                break

            parts.append(chunk)
            total_len += len(chunk)

        return "\n\n---\n\n".join(parts)


# 全局单例
_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
