"""
HyDE (Hypothetical Document Embedding) 问题假设生成器

原理（参考掌柜智库项目）:
  用户短查询 → LLM 生成"假设性答案文档" → 拼接到原查询 → 向量检索

为什么有效:
  - 短查询语义稀疏，直接检索可能匹配不准确
  - 假设文档更接近知识库中实际文档的语言风格和长度
  - 生成的文档包含更多关键词和上下文，检索信号更丰富

使用方式:
  generator = HyDEGenerator(llm)
  hyde_doc = await generator.generate("如何设置烫金参数？")
  # → "烫金参数设置需要在操作面板进入【转印参数】→【区域设置】..."
"""

from typing import Optional

from rag_agent.config import rag_config

# HyDE 提示词：让 LLM 生成专业的假设性回答
HYDE_PROMPT_TEMPLATE = """你是一位专业领域的专家。请根据以下用户问题，生成一段简洁专业的回答。

用户问题: {query}

要求:
1. 回答要简洁明了，包含核心信息即可
2. 假设你是该领域的专家，提供专业的解释
3. 不要使用"假设"、"可能"等不确定的词汇
4. 保持回答与查询主题高度相关
5. 使用中文回答且不超过300字
"""


class HyDEGenerator:
    """
    假设文档生成器

    将用户问题扩展为假设性答案文本，增强语义检索效果。
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain ChatModel 实例。不传则延迟导入项目 LLM。
        """
        self._llm = llm
        self.enabled = rag_config.hyde_enabled
        self.max_tokens = rag_config.hyde_max_tokens

    @property
    def llm(self):
        if self._llm is None:
            # 延迟导入项目 LLM
            from agent.llm import model
            self._llm = model
        return self._llm

    def generate_prompt(self, query: str) -> str:
        """构建 HyDE 提示词"""
        return HYDE_PROMPT_TEMPLATE.format(query=query)

    async def agenerate(self, query: str) -> Optional[str]:
        """
        异步生成假设文档。

        Args:
            query: 用户原始问题

        Returns:
            假设文档文本；失败或禁用时返回 None
        """
        if not self.enabled:
            return None

        try:
            prompt = self.generate_prompt(query)
            response = await self.llm.ainvoke(prompt)
            hyde_doc = response.content.strip()
            if hyde_doc:
                print(f"[HyDE] 生成假设文档: {len(hyde_doc)}字符")
                return hyde_doc
        except Exception as e:
            print(f"[HyDE] 生成失败: {e}")

        return None

    def generate(self, query: str) -> Optional[str]:
        """
        同步生成假设文档。

        Args:
            query: 用户原始问题

        Returns:
            假设文档文本；失败或禁用时返回 None
        """
        if not self.enabled:
            return None

        try:
            prompt = self.generate_prompt(query)
            response = self.llm.invoke(prompt)
            hyde_doc = response.content.strip()
            if hyde_doc:
                print(f"[HyDE] 生成假设文档: {len(hyde_doc)}字符")
                return hyde_doc
        except Exception as e:
            print(f"[HyDE] 生成失败: {e}")

        return None

    def enrich_query(self, query: str, hyde_doc: Optional[str] = None) -> str:
        """
        拼接原查询与假设文档，形成富语义检索文本。

        Args:
            query: 原始查询
            hyde_doc: HyDE 生成的假设文档（可选）

        Returns:
            增强后的查询文本
        """
        if hyde_doc and self.enabled:
            return f"{query}\n{hyde_doc}"
        return query


# 全局单例
_hyde_generator: Optional[HyDEGenerator] = None


def get_hyde_generator(llm=None) -> HyDEGenerator:
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator(llm=llm)
    return _hyde_generator
