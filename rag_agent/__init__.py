"""
RAG Agent 包 - 本地知识库检索系统

核心特性:
  - 父子文档关联切分 (Markdown标题优先, 子块512/父块2048)
  - HyDE 问题假设增强 (LLM生成假设答案→增强检索语义)
  - BGE-M3 向量化 (1024维稠密向量)
  - Qdrant 向量存储 (父子双集合架构)
  - BGE Reranker v2-m3 重排序 (Cross-Encoder精排 + 动态TopK)
  - MD5 文件去重 (增量更新, 避免重复处理)

依赖:
  pip install FlagEmbedding qdrant-client sentence-transformers
       pypdf python-docx langchain-text-splitters
"""
