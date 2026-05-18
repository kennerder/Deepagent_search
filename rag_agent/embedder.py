"""
向量化模块：BGE-M3 模型封装

BGE-M3 特点:
  - 多语言支持 (中英文)
  - 1024维稠密向量 (Dense) + 稀疏向量 (Sparse Lexical Weights)
  - 单次推理同时产出两种向量，兼顾语义匹配和关键词匹配

后端选择:
  - FlagEmbedding (原生): 支持 Dense + Sparse 双路输出
  - sentence-transformers (备选): 仅 Dense，兼容性好
"""

import os
from typing import List, Optional

# 设置 HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from rag_agent.config import rag_config


class BGEEmbedder:
    """
    BGE-M3 向量化封装（单例模式）

    使用方式:
        embedder = get_embedder()
        dense_vec = embedder.encode("文本")  # 稠密向量 (1024维)
        # dense, sparse = embedder.encode_both("文本")  # 双路输出
    """

    _instance: Optional["BGEEmbedder"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        """延迟加载模型（首次调用时加载）"""
        if self._model is not None:
            return

        model_name = rag_config.embedding_model_name
        device = rag_config.embedding_device

        # 优先尝试 FlagEmbedding (支持 Dense + Sparse)
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                model_name_or_path=model_name,
                use_fp16=(device != "cpu"),
                device=device,
            )
            self._backend = "flagembedding"
            print(f"[Embedder] BGE-M3 loaded via FlagEmbedding (device={device})")
        except ImportError:
            # 降级到 sentence-transformers (仅 Dense)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._backend = "sentence_transformers"
            print(f"[Embedder] BGE-M3 loaded via sentence-transformers (device={device})")

    @property
    def model(self):
        self._load_model()
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        将文本列表编码为稠密向量列表。

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            向量列表，每个向量为1024维 float 列表
        """
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()

        if self._backend == "flagembedding":
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                result = self._model.encode(batch, batch_size=batch_size)
                all_embeddings.extend(result["dense_vecs"].tolist())
            return all_embeddings
        else:
            # sentence-transformers
            return self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
            ).tolist()

    def encode_both(self, text: str) -> dict:
        """
        生成稠密 + 稀疏双路向量（仅 FlagEmbedding 后端支持）。

        Returns:
            {"dense": [[1024 floats]], "sparse": [Lexical weights dict]}
        """
        self._load_model()
        if self._backend != "flagembedding":
            raise RuntimeError("Sparse embedding 需要 FlagEmbedding 后端")

        result = self._model.encode([text], return_dense=True, return_sparse=True)
        return {
            "dense": result["dense_vecs"].tolist(),
            "sparse": result.get("lexical_weights", []),
        }

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """编码查询文本（与 encode 相同，保留语义接口）"""
        return self.encode(queries)

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """编码文档文本（与 encode 相同，保留语义接口）"""
        return self.encode(documents)


# 全局单例
_embedder: Optional[BGEEmbedder] = None


def get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder
