"""
RAG Agent 配置模块

技术栈:
  - 向量模型: BAAI/bge-m3 (1024维, 中英文, 稠密+稀疏双路)
  - 重排模型: BAAI/bge-reranker-v2-m3 (多语言, FlagEmbedding)
  - 向量数据库: Qdrant
  - LLM: 复用项目 LLM 配置 (qwen-max)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RAGConfig:
    """RAG Agent 全局配置"""

    # --- 文档目录 ---
    # 固定目录：程序从此目录加载文档
    documents_dir: str = "data/documents"
    # 去重记录文件
    dedup_record_file: str = "data/processed_files.json"

    # --- 切分配置 ---
    # 子块最大字符数 (用于检索)
    child_chunk_size: int = 512
    # 子块重叠字符数
    child_chunk_overlap: int = 50
    # 父块最大字符数 (返回上下文)
    parent_chunk_size: int = 2048
    # 父块重叠字符数
    parent_chunk_overlap: int = 100
    # Markdown 标题切分优先
    use_markdown_headers: bool = True
    # 短块合并阈值
    min_chunk_length: int = 100

    # --- 向量模型 ---
    # BGE-M3 模型名 (HuggingFace / 本地路径)
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    # Qdrant 集合配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    # 子块集合 (用于检索 - 高精度匹配)
    child_collection_name: str = "rag_child_chunks"
    # 父块集合 (用于返回完整上下文)
    parent_collection_name: str = "rag_parent_chunks"
    # 向量维度 (bge-m3: 1024)
    vector_size: int = 1024

    # --- 重排模型 ---
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_use_fp16: bool = False

    # --- 检索配置 ---
    # 向量检索召回候选数 (重排前)
    retrieval_top_k: int = 20
    # 重排后返回数量
    rerank_top_k: int = 5
    # 动态TopK断崖阈值 (相对)
    rerank_gap_ratio: float = 0.25
    # 动态TopK断崖阈值 (绝对)
    rerank_gap_abs: float = 0.5
    # 重排最小保留数
    rerank_min_topk: int = 1
    # 重排最大保留数
    rerank_max_topk: int = 10

    # --- HyDE 配置 ---
    # 是否启用 HyDE
    hyde_enabled: bool = True
    # HyDE 生成的最大 token 数
    hyde_max_tokens: int = 300

    # --- 去重配置 ---
    # 文件去重方式: "md5" | "path" | "both"
    dedup_method: str = "md5"


# 项目根目录 (rag_agent → 项目根)
PROJECT_ROOT = Path(__file__).parent.parent

# 全局配置实例
rag_config = RAGConfig(
    documents_dir=str(PROJECT_ROOT / "data" / "documents"),
    dedup_record_file=str(PROJECT_ROOT / "data" / "processed_files.json"),
)
