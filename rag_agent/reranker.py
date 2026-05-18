"""
BGE Reranker 重排序模块

模型: BAAI/bge-reranker-v2-m3
  - 多语言支持 (中英文)
  - 基于交叉编码器 (Cross-Encoder) 对 <query, document> 对做精细打分
  - 相比向量检索的余弦相似度，重排模型能更准确地判断语义相关性

动态 TopK 策略（参考掌柜智库项目）:
  不是固定取前 N 名，而是检测"分数断崖"自动截断：
  前5名分数: [0.95, 0.93, 0.89, 0.77, 0.71]
  断崖检测: 0.89 vs 0.77 差距 > 阈值 → 只取前3名

使用方式:
  reranker = get_reranker()
  results = reranker.rerank("用户问题", [{"content": "文档1"}, ...])
  # → [{"content": "文档1", "rerank_score": 0.95}, ...]
"""

from typing import List, Dict, Optional

from rag_agent.config import rag_config


class BGEReranker:
    """
    BGE Reranker 封装（单例模式）

    使用 FlagEmbedding.FlagReranker 加载 BAAI/bge-reranker-v2-m3。
    """

    _instance: Optional["BGEReranker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        if self._model is not None:
            return

        model_name = rag_config.reranker_model_name
        device = rag_config.reranker_device

        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                model_name_or_path=model_name,
                device=device,
                use_fp16=rag_config.reranker_use_fp16,
            )
            print(f"[Reranker] Model loaded: {model_name} (device={device})")
        except ImportError:
            print("[Reranker] FlagEmbedding not installed. "
                  "Install with: pip install FlagEmbedding")
            self._model = None
        except Exception as e:
            print(f"[Reranker] Failed to load model: {e}")
            self._model = None

    @property
    def model(self):
        self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        对检索结果进行重排序。

        Args:
            query: 用户查询
            documents: 检索结果列表 [{"content": str, ...}, ...]
            top_k: 返回数量（默认使用动态 TopK）

        Returns:
            重排后的文档列表，每项附加 "rerank_score" 字段
        """
        if not documents or not query:
            return documents

        if self._model is None:
            # 模型未加载，返回原始结果（标记为未重排）
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0)
            return documents

        # 构造 <query, document> 对
        pairs = [[query, doc.get("content", "")] for doc in documents]

        # 模型打分
        scores = self._model.compute_score(pairs, normalize=True)
        # 确保 scores 是列表
        if not isinstance(scores, list):
            scores = [scores]

        # 附加分数
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # 排序
        documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # 动态 TopK
        if top_k is None:
            top_k = self._dynamic_topk(documents)

        return documents[:top_k]

    def _dynamic_topk(self, documents: List[Dict]) -> int:
        """
        动态 TopK：检测分数断崖自动截断。

        规则:
        1. 计算相邻文档的分数落差
        2. 若落差超过阈值（相对或绝对），在该处截断
        3. 最少保留 min_topk，最多保留 max_topk
        """
        if len(documents) <= 1:
            return len(documents)

        scores = [doc.get("rerank_score", 0) for doc in documents]
        gap_ratio = rag_config.rerank_gap_ratio
        gap_abs = rag_config.rerank_gap_abs
        min_k = rag_config.rerank_min_topk
        max_k = rag_config.rerank_max_topk

        cut = len(documents)
        for i in range(min_k, min(len(scores), max_k)):
            prev_score = scores[i - 1]
            curr_score = scores[i]

            # 相对断崖: (prev - curr) / prev > gap_ratio
            # 绝对断崖: (prev - curr) > gap_abs
            if prev_score > 0:
                relative_gap = (prev_score - curr_score) / prev_score
                if relative_gap > gap_ratio or (prev_score - curr_score) > gap_abs:
                    cut = i
                    break

        return min(cut, max_k, len(documents))


# 全局单例
_reranker: Optional[BGEReranker] = None


def get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker
