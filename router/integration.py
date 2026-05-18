"""
BERT 路由集成模块

将 BERT 意图分类器集成到 main_agent 的执行流程中。

集成方式:
  - 在 run_deep_agent 中，先通过 BERT 判断需要调用哪些子代理
  - 多子代理时，LLM 拆解 query 为各子代理的精准子任务
  - 将判断结果写入提示词，告知 LLM "本次任务建议调用以下子代理"
  - LLM 仍负责工具调用和具体执行编排
  - BERT 低置信度时 fallback 到纯 LLM 路由

路由流程:
  raw_query → (可选) LLM 指代消解改写 → BERT 多标签分类
  → 命中多个子代理: LLM 拆解 query → 校验 → 注入拆解子任务到 prompt
  → 命中单个子代理: 直接注入路由建议
  → 低置信度: 走 LLM fallback, 不加路由约束

防幻觉策略:
  Layer 1: 严格 Prompt 约束 (只拆不增，禁止编造)
  Layer 2: 实体覆盖率校验 (拆解后关键实体需覆盖原 query 的 50%+)
  Layer 3: 校验失败 → 回退传原 query (当前行为，零风险)

使用方式:
  from router.integration import RouteRouter
  router = RouteRouter()
  route_result = await router.route("查布洛芬库存并搜索最新市场行情")
  # → labels: ["db_query", "web_search"], sub_tasks: {"db_query": "...", "web_search": "..."}
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from router.config import (
    router_config, SUB_AGENT_LABELS,
    LABEL_CN_MAP, LABEL_SUB_AGENT_MAP,
)
from router.classifier import get_classifier


@dataclass
class RouteResult:
    """路由结果"""
    # 命中的标签列表
    labels: List[str] = field(default_factory=list)
    # 对应的置信度
    scores: List[float] = field(default_factory=list)
    # 路由模式: "bert" | "llm_fallback" | "not_loaded"
    mode: str = "not_loaded"
    # 所有标签的分数 (调试用)
    all_scores: List[Dict] = field(default_factory=list)
    # 拆解后的子任务映射: {label: sub_query} (多子代理时有效)
    sub_tasks: Dict[str, str] = field(default_factory=dict)
    # 路由建议提示词 (可直接注入到 system_prompt)
    route_hint: str = ""


# ===================== Query 拆解 Prompt =====================

DECOMPOSE_SYSTEM_PROMPT = """你是一个任务拆解专家。你的唯一职责是将用户的复合查询严格按子代理职责拆解为子任务。

子代理及其职责:
- 数据库查询助手(db_query): 负责查询结构化数据库，如药品库存、销售数据、药品信息等
- 网络搜索助手(web_search): 负责搜索互联网公开信息，如新闻、政策、市场行情等
- 知识库助手(kb_search): 负责检索企业内部私有文档，如规章制度、操作手册、质量文件等

严格规则 (违反任何一条即为失败):
1. 【只拆不增】只能从原始查询中提取词语和意图，绝对禁止编造、推测或添加原始查询中没有的信息
2. 【完整覆盖】原始查询中的所有意图必须被覆盖，不能遗漏任何一个
3. 【独立可执行】每个子任务是独立完整的自然语言句子，可被对应子代理直接理解执行
4. 【职责对齐】每个子任务只包含对应子代理职责范围内的内容，不要把搜索任务分给数据库助手
5. 【去噪】从原始查询中移除不属于某子代理职责的噪音部分

输出格式: 每行一个子任务，格式为 "标签: 子任务内容"
示例输入: "查布洛芬库存并搜索最新市场行情"
示例输出:
db_query: 查询布洛芬的库存数量
web_search: 搜索布洛芬的最新市场行情"""


class RouteRouter:
    """
    BERT 路由调度器

    负责:
      1. 查询改写 (指代消解, 可选)
      2. BERT 多标签意图分类
      3. Query 拆解 (多子代理时) + 防幻觉校验
      4. 置信度判断 + fallback 策略
      5. 生成路由提示词注入
    """

    def __init__(self):
        self.classifier = get_classifier()
        self._loaded = False

    def ensure_loaded(self):
        """确保模型已加载"""
        if not self._loaded:
            try:
                self.classifier.load()
                self._loaded = True
            except Exception as e:
                print(f"[Router] 模型加载失败，将使用 LLM fallback: {e}")
                self._loaded = False

    async def route(
        self,
        query: str,
        thread_id: str = None,
        use_rewrite: bool = None,
    ) -> RouteResult:
        """
        执行路由决策。

        Args:
            query: 用户查询 (已改写或原始)
            thread_id: 会话 ID (用于加载对话历史做指代消解)
            use_rewrite: 是否先 LLM 改写 (默认使用配置值)

        Returns:
            RouteResult 路由结果
        """
        use_rewrite = use_rewrite if use_rewrite is not None else router_config.enable_query_rewrite

        # Step 0: 加载模型
        self.ensure_loaded()

        # Step 1: 可选 - LLM 指代消解改写 (利用对话历史)
        resolved_query = query
        if use_rewrite:
            resolved_query = await self._rewrite_query(query, thread_id=thread_id)

        # Step 2: BERT 分类 + 可选拆解
        if self._loaded:
            return await self._bert_route(resolved_query)
        else:
            return self._llm_fallback_route(resolved_query)

    def route_sync(self, query: str) -> RouteResult:
        """同步路由（不包含 LLM 改写和拆解，仅 BERT 分类）"""
        self.ensure_loaded()
        if self._loaded:
            return self._bert_route_sync(query)
        else:
            return self._llm_fallback_route(query)

    # ===================== BERT 路由 =====================

    async def _bert_route(self, query: str) -> RouteResult:
        """BERT 分类 + 置信度判断 + 多子代理 Query 拆解 + 无匹配检测"""
        # 单次推理，获取所有标签分数
        all_scores = self.classifier.predict(query, return_all_scores=True)
        # all_scores: [{"label": "db_query", "score": 0.87}, {"label": "web_search", "score": 0.12}, ...]
        scores_by_label = {s["label"]: s["score"] for s in all_scores}
        max_score = max(scores_by_label.values()) if scores_by_label else 0.0

        if router_config.multi_label:
            threshold = router_config.multi_label_threshold
            hit_labels = [s["label"] for s in all_scores if s["score"] >= threshold]
            hit_scores = [s["score"] for s in all_scores if s["score"] >= threshold]

            if hit_labels:
                # 命中 → 正常路由
                sub_tasks: Dict[str, str] = {}
                if len(hit_labels) > 1:
                    sub_tasks = await self._decompose_query(query, hit_labels)

                return RouteResult(
                    labels=hit_labels,
                    scores=hit_scores,
                    mode="bert",
                    all_scores=all_scores,
                    sub_tasks=sub_tasks,
                    route_hint=self._build_hint(hit_labels, hit_scores, sub_tasks),
                )

            # 未命中 → 区分"完全无关"和"模棱两可"
            if max_score < router_config.none_threshold:
                print(f"[Router] 判定为无关查询 (max_score={max_score:.3f} < {router_config.none_threshold})")
                return RouteResult(
                    labels=[],
                    scores=[],
                    mode="none",
                    all_scores=all_scores,
                    route_hint=self._build_none_hint(scores_by_label),
                )
            else:
                print(f"[Router] 置信度不足 (max_score={max_score:.3f}), 走 LLM fallback")
                return RouteResult(
                    labels=[],
                    scores=[],
                    mode="llm_fallback",
                    all_scores=all_scores,
                    route_hint="",
                )
        else:
            top = all_scores[0] if all_scores else {}
            if top.get("score", 0) >= router_config.single_label_threshold:
                return RouteResult(
                    labels=[top["label"]],
                    scores=[top["score"]],
                    mode="bert",
                    all_scores=all_scores,
                    route_hint=self._build_hint([top["label"]], [top["score"]], {}),
                )

        return RouteResult(
            labels=[],
            scores=[],
            mode="llm_fallback",
            all_scores=all_scores,
            route_hint="",
        )

    def _bert_route_sync(self, query: str) -> RouteResult:
        """同步版：仅 BERT 分类，不做拆解，支持无匹配检测"""
        all_scores = self.classifier.predict(query, return_all_scores=True)
        scores_by_label = {s["label"]: s["score"] for s in all_scores}

        if router_config.multi_label:
            threshold = router_config.multi_label_threshold
            hit_labels = [s["label"] for s in all_scores if s["score"] >= threshold]
            hit_scores = [s["score"] for s in all_scores if s["score"] >= threshold]
            max_score = max(scores_by_label.values()) if scores_by_label else 0.0

            if hit_labels:
                return RouteResult(
                    labels=hit_labels,
                    scores=hit_scores,
                    mode="bert",
                    all_scores=all_scores,
                    route_hint=self._build_hint(hit_labels, hit_scores, {}),
                )

            if max_score < router_config.none_threshold:
                return RouteResult(
                    labels=[],
                    scores=[],
                    mode="none",
                    all_scores=all_scores,
                    route_hint=self._build_none_hint(scores_by_label),
                )
        return self._llm_fallback_route(query)

    # ===================== LLM Fallback =====================

    def _llm_fallback_route(self, query: str) -> RouteResult:
        """LLM fallback: 让 LLM 自行判断，不加路由约束"""
        return RouteResult(
            labels=[],
            scores=[],
            mode="llm_fallback",
            route_hint="",
        )

    # ===================== 查询改写 =====================

    async def _rewrite_query(self, query: str, thread_id: str = None) -> str:
        """
        LLM 指代消解改写。利用对话历史补全省略的主语和上下文。

        只做指代消解和省略补充，不做意图分类。
        改写后的 query 送入 BERT 分类。

        例如:
          历史: 用户问"布洛芬库存" → Agent 答复了库存情况
          当前: "那销售额呢？" → "布洛芬的销售额是多少？"

        策略:
          - query >= 10 字且无历史 → 跳过改写（大概率已自包含）
          - query < 10 字或有历史 → 结合历史做指代消解
        """
        # 加载对话历史
        history_text = ""
        has_history = False
        if thread_id:
            try:
                from api.memory import memory_manager
                raw_history = await memory_manager.get_history(thread_id, max_turns=3)
                if raw_history:
                    has_history = True
                    history_text = "对话历史:\n"
                    for msg in raw_history:
                        role = "用户" if msg["role"] == "user" else "助手"
                        content = msg["content"][:300]
                        history_text += f"  {role}: {content}\n"
            except Exception:
                pass

        # 自包含的短 query 且无历史 → 跳过
        if len(query) >= 10 and not has_history:
            return query

        # 需要改写
        try:
            from agent.llm import model

            if has_history:
                prompt = (
                    f"{history_text}\n"
                    "基于以上对话历史，将以下用户问题的指代消解为完整、自包含的问题。\n"
                    "只补充上下文中明确的主语和对象，不要编造任何信息：\n"
                    f"用户问题：{query}\n"
                    "改写后："
                )
            else:
                prompt = (
                    "请将以下用户问题改写为完整、自包含的问题（补充缺失的主语和上下文）：\n"
                    f"用户问题：{query}\n"
                    "改写后："
                )

            response = await model.ainvoke(prompt)
            rewritten = response.content.strip()
            if rewritten and rewritten != query:
                print(f"[Router] 指代消解: '{query}' → '{rewritten}'")
                return rewritten
            return query
        except Exception as e:
            print(f"[Router] 改写失败: {e}")
            return query

    # ===================== Query 拆解 =====================

    async def _decompose_query(
        self, query: str, labels: List[str]
    ) -> Dict[str, str]:
        """
        LLM 拆解原始 query 为各子代理的精准子任务。

        防幻觉三层防御:
          1. 严格 Prompt 约束 (只拆不增)
          2. 实体覆盖率校验
          3. 校验失败 → 回退空 dict (兜底: 传原 query)

        Args:
            query: 原始查询
            labels: BERT 命中的子代理标签列表

        Returns:
            {label: sub_query} 映射，校验失败返回空 dict
        """
        if len(labels) <= 1:
            return {}

        try:
            from agent.llm import model
            from langchain_core.messages import HumanMessage, SystemMessage

            label_list = ", ".join(
                f"{label}({LABEL_CN_MAP.get(label, label)})" for label in labels
            )

            user_prompt = (
                f'原始查询: """{query}"""\n\n'
                f"需要拆解到的子代理: {label_list}\n\n"
                f"请按格式拆解:"
            )

            response = await model.ainvoke([
                SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            raw_output = response.content.strip()
            sub_tasks = self._parse_decompose_output(raw_output, labels)

            # 校验
            if self._validate_decomposition(query, sub_tasks, labels):
                print(f"[Router] Query 拆解成功: {sub_tasks}")
                return sub_tasks
            else:
                print(f"[Router] Query 拆解校验失败，回退为原始 query")
                return {}

        except Exception as e:
            print(f"[Router] Query 拆解异常: {e}，回退为原始 query")
            return {}

    def _parse_decompose_output(
        self, raw: str, labels: List[str]
    ) -> Dict[str, str]:
        """
        解析 LLM 拆解输出。

        输入格式: "db_query: 查询布洛芬库存\nweb_search: 搜索布洛芬行情"
        """
        sub_tasks: Dict[str, str] = {}
        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # 匹配 "label: content" 或 "label： content"
            match = re.match(r'^(\w+)\s*[:：]\s*(.+)$', line)
            if match:
                label = match.group(1).strip()
                content = match.group(2).strip()
                if label in labels and content:
                    sub_tasks[label] = content

        return sub_tasks

    # ===================== 拆解校验 =====================

    def _validate_decomposition(
        self,
        query: str,
        sub_tasks: Dict[str, str],
        labels: List[str],
    ) -> bool:
        """
        校验 LLM 拆解结果，防止幻觉。

        检查项:
          1. 数量一致: 拆解出的子任务数 == BERT 命中标签数
          2. 非空: 每个子任务不为空
          3. 实体覆盖: 拆解后关键实体覆盖原 query 的 50%+
          4. 无编造: 新增实体不超过原 query 实体的 50%
        """
        # Check 1: 数量一致
        if set(sub_tasks.keys()) != set(labels):
            print(f"  [校验失败] 标签不匹配: 拆解={set(sub_tasks.keys())}, 期望={set(labels)}")
            return False

        # Check 2: 非空
        if any(not v.strip() for v in sub_tasks.values()):
            print(f"  [校验失败] 存在空子任务")
            return False

        # Check 3 & 4: 实体覆盖 + 无编造
        original_entities = self._extract_key_entities(query)
        if not original_entities:
            return True  # 原始 query 太短，跳过实体校验

        decomposed_text = ' '.join(sub_tasks.values())
        decomposed_entities = self._extract_key_entities(decomposed_text)

        # 覆盖率: 原 query 中多少实体在拆解中被保留
        covered = original_entities & decomposed_entities
        coverage = len(covered) / len(original_entities)

        # 编造率: 拆解中多少实体是原 query 没有的
        new_entities = decomposed_entities - original_entities
        fabrication_rate = len(new_entities) / max(len(original_entities), 1)

        print(f"  [校验] 实体覆盖={coverage:.2f} (原{len(original_entities)}个/保留{len(covered)}个)"
              f" 编造率={fabrication_rate:.2f} (新{len(new_entities)}个)")

        if coverage < 0.5:
            print(f"  [校验失败] 实体覆盖率不足: {coverage:.2f} < 0.5")
            return False

        if fabrication_rate > 0.5:
            print(f"  [校验失败] 编造率过高: {fabrication_rate:.2f} > 0.5")
            return False

        return True

    def _extract_key_entities(self, text: str) -> set:
        """
        提取文本中的关键实体（用于校验覆盖/编造）。

        使用 jieba TF-IDF 提取核心词 + 名词过滤。
        """
        try:
            import jieba.analyse
            # TF-IDF 关键词 (TopK=8)
            keywords = set(jieba.analyse.extract_tags(text, topK=8, withWeight=False))
            # 补充: 长度 >= 2 的 jieba 分词结果
            words = set(w for w in jieba.lcut(text) if len(w) >= 2)
            return keywords | words
        except Exception:
            # jieba 不可用时回退为简单字符级 n-gram
            chars = text.replace(' ', '')
            bigrams = {chars[i:i+2] for i in range(len(chars)-1)}
            return bigrams

    # ===================== 提示词构建 =====================

    def _build_hint(
        self,
        labels: List[str],
        scores: List[float],
        sub_tasks: Dict[str, str] = None,
    ) -> str:
        """
        构建路由建议提示词，注入到主代理的 system_prompt。

        包含拆解后的子任务（如有），指导主 Agent LLM 精准调用子代理。

        Args:
            labels: 命中标签列表
            scores: 对应分数列表
            sub_tasks: 拆解后的子任务 {label: sub_query}，空则只输出标签建议

        Returns:
            路由建议文本
        """
        if not labels:
            return ""

        sub_tasks = sub_tasks or {}
        has_sub_tasks = len(sub_tasks) > 0

        lines = ["【路由建议】根据意图分类，本次任务建议调用以下子代理:"]
        for label, score in zip(labels, scores):
            cn_name = LABEL_CN_MAP.get(label, label)
            lines.append(f"  - {cn_name} (置信度: {score:.2f})")
            if has_sub_tasks and label in sub_tasks:
                lines.append(f"    子任务: \"{sub_tasks[label]}\"")

        if has_sub_tasks:
            lines.append("")
            lines.append("请使用 task 工具分别调用上述子代理，为每个子代理传递对应的「子任务」作为 description。")
        else:
            lines.append("请优先调用上述子代理完成任务。")

        return "\n".join(lines)

    def _build_none_hint(self, scores_by_label: Dict[str, float]) -> str:
        """
        构建"无匹配"提示词——告知主 Agent 本次查询不需要任何子代理。

        Args:
            scores_by_label: {label: score} 映射

        Returns:
            指示主 Agent 直接作答的提示词
        """
        sorted_scores = sorted(scores_by_label.items(), key=lambda x: x[1], reverse=True)
        score_detail = ", ".join(
            f"{LABEL_CN_MAP.get(l, l)}={s:.2f}" for l, s in sorted_scores
        )

        return (
            f"【路由建议】BERT 判定本次查询与三个子代理均不相关 "
            f"(得分: {score_detail})。\n"
            f"请勿调用任何子代理（task 工具），由主 Agent 直接回答即可。\n"
            f"如用户只是打招呼或闲聊，简单回应即可。"
        )

    # ===================== 状态 =====================

    def get_status(self) -> Dict:
        """获取路由模块状态"""
        return {
            "model_loaded": self._loaded,
            "model_info": self.classifier.get_model_info() if self._loaded else {},
            "config": {
                "multi_label": router_config.multi_label,
                "threshold": router_config.single_label_threshold,
                "fallback": router_config.low_confidence_fallback,
                "query_rewrite": router_config.enable_query_rewrite,
            },
        }


# 全局单例
_router: Optional[RouteRouter] = None


def get_router() -> RouteRouter:
    global _router
    if _router is None:
        _router = RouteRouter()
    return _router
