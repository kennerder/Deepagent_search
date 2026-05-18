"""
BERT 子代理路由模块

功能: 用 BERT 分类替代 LLM 路由，提高子代理调度的可靠性。

模型: hfl/chinese-roberta-wwm-ext (哈工大, 110M参数, 全词掩码)

路由流程:
  原始 query → (可选) LLM 指代消解 → BERT 分类
  → 高置信度: 注入路由建议到 prompt
  → 低置信度: LLM fallback

使用:
  from router.integration import get_router
  router = get_router()
  result = await router.route("查询布洛芬库存")
"""
