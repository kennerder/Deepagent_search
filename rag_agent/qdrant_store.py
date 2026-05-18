"""
Qdrant 向量存储模块

支持父子双集合架构:
  - child_collection: 子块索引（检索入口，高精度匹配）
  - parent_collection: 父块索引（上下文返回，完整语义）

操作:
  - 建库 (create_collections)
  - 批量写入 (upsert_parents, upsert_children)
  - 向量检索 (search)
  - 根据子块ID追溯父块 (get_parents_by_child_ids)
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
)

from rag_agent.config import rag_config


class QdrantStore:
    """Qdrant 向量存储，管理父子双集合"""

    def __init__(self):
        self.client = QdrantClient(
            host=rag_config.qdrant_host,
            port=rag_config.qdrant_port,
        )
        self.child_collection = rag_config.child_collection_name
        self.parent_collection = rag_config.parent_collection_name
        self.vector_size = rag_config.vector_size

    # ========================= 集合管理 =========================

    def collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def create_collections(self, force: bool = False):
        """创建父子集合"""
        for name in [self.child_collection, self.parent_collection]:
            if force and self.collection_exists(name):
                self.client.delete_collection(name)
            if not self.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

    # ========================= 批量写入 =========================

    def upsert_parents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """批量写入父块"""
        if not chunks:
            return
        points = [
            PointStruct(
                id=chunk["id"],
                vector=emb,
                payload={
                    "content": chunk["content"],
                    "title": chunk.get("title", ""),
                    "file_name": chunk.get("file_name", ""),
                    "section_title": chunk.get("section_title", ""),
                    "part": chunk.get("part", 0),
                    "chunk_type": "parent",
                }
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.parent_collection, points=points)

    def upsert_children(self, chunks: List[Dict], embeddings: List[List[float]]):
        """批量写入子块"""
        if not chunks:
            return
        points = [
            PointStruct(
                id=chunk["id"],
                vector=emb,
                payload={
                    "content": chunk["content"],
                    "parent_id": chunk.get("parent_id", ""),
                    "title": chunk.get("title", ""),
                    "file_name": chunk.get("file_name", ""),
                    "parent_title": chunk.get("parent_title", ""),
                    "part": chunk.get("part", 0),
                    "chunk_type": "child",
                }
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.child_collection, points=points)

    # ========================= 检索 =========================

    def search(
        self,
        query_vector: List[float],
        top_k: int = None,
        collection: Optional[str] = None,
    ) -> List[Dict]:
        """
        向量检索。

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            collection: 搜索的集合名（默认子块集合）

        Returns:
            [{"id", "score", "payload": {...}}, ...]
        """
        top_k = top_k or rag_config.retrieval_top_k
        collection = collection or self.child_collection

        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload or {},
            }
            for point in results.points
        ]

    def get_parents_by_child_ids(self, child_results: List[Dict]) -> List[Dict]:
        """
        根据子块检索结果，追溯对应的父块。

        核心逻辑：子块命中 → 查 parent_id → 返回父块完整内容
        """
        parent_ids = list(set(
            r.get("payload", {}).get("parent_id", "")
            for r in child_results
            if r.get("payload", {}).get("parent_id")
        ))

        if not parent_ids:
            return []

        # 批量获取父块
        retrieved = self.client.retrieve(
            collection_name=self.parent_collection,
            ids=parent_ids,
        )

        id_to_content = {}
        for point in retrieved:
            payload = point.payload or {}
            id_to_content[point.id] = {
                "id": point.id,
                "content": payload.get("content", ""),
                "title": payload.get("title", ""),
                "file_name": payload.get("file_name", ""),
            }

        # 保持子块的排序（去重）
        seen = set()
        ordered_parents = []
        for child in child_results:
            pid = child.get("payload", {}).get("parent_id", "")
            if pid and pid not in seen and pid in id_to_content:
                seen.add(pid)
                ordered_parents.append({
                    **id_to_content[pid],
                    "child_score": child.get("score"),
                })

        return ordered_parents

    def get_collection_info(self) -> Dict[str, int]:
        """获取集合统计信息"""
        info = {}
        for name in [self.child_collection, self.parent_collection]:
            try:
                info[name] = self.client.count(name).count
            except Exception:
                info[name] = 0
        return info


# 全局单例
_qdrant_store: Optional[QdrantStore] = None


def get_qdrant_store() -> QdrantStore:
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantStore()
    return _qdrant_store
