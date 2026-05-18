"""
DeepAgents Search 统一评测模块

覆盖三项评测:
  - router:  BERT 子代理路由准确率 (多标签 Exact Match + 每标签 F1)
  - rag:     RAG 管线评测 (RAGAS: Faithfulness / Relevancy / Precision / Recall)
  - sql:     SQL Agent 评测 (BIRD-style EX: 执行准确率 + 错误分类)

使用:
  python -m evaluation.cli --all              # 全量评测
  python -m evaluation.cli --router           # 仅路由
  python -m evaluation.cli --rag              # 仅 RAG
  python -m evaluation.cli --sql              # 仅 SQL
"""
