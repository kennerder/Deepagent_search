"""
多角度深度搜索工具 (Multi-Angle Deep Search)

核心设计:
  LLM 改写 → 硬编码 3~5 轮并行搜索 → URL 去重 → 高密度上下文聚合

设计目标:
  通过多角度、多关键词的广域搜索，构建高密度全局上下文，有效降低单一信息源偏见。
  硬编码规则强制执行 3~5 次搜索，不由 LLM 决定搜索次数，确保覆盖广度。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from api.monitor import monitor


# ===================== 查询改写 =====================

REWRITE_SYSTEM_PROMPT = """你是一个专业的搜索查询改写专家。你的任务是将用户的原始查询改写为 3~5 个不同角度的搜索查询词。

改写原则:
1. 角度拆解: 从定义、原理、应用、案例、对比、趋势、争议等维度拆解
2. 关键词变换: 同一概念使用不同表述（同义词、中英文、缩写/全称）
3. 范围收放: 既有宽泛搜索也有精准搜索，兼顾广度和深度
4. 立场多元: 涉及观点类问题时，从不同立场/学派/地区角度搜索
5. 时效兼顾: 涉及动态信息时，加入"最新""2025""趋势"等时效词

输出格式: 每个搜索查询词独占一行，不要编号、不要引号、不要额外解释。"""


def _rewrite_query(query: str, llm) -> List[str]:
    """
    使用 LLM 将原始查询改写为 3~5 个不同角度的搜索词。

    Args:
        query: 用户原始查询
        llm: LLM 实例

    Returns:
        改写后的搜索词列表 (3~5 个)
    """
    user_prompt = f'原始查询: """{query}"""\n\n请输出 3~5 个不同角度的搜索词:'

    try:
        response = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        # 解析: 按行分割，过滤空行和过短行
        lines = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
        queries = [line.lstrip('0123456789. -·•') for line in lines]
        queries = [q for q in queries if len(q) >= 3]

        # 硬编码规则: 确保 3~5 个
        if len(queries) > 5:
            queries = queries[:5]
        elif len(queries) < 3:
            # 不足 3 个时，补充原始查询
            if query not in queries:
                queries.append(query)
            # 尝试进一步拆解
            while len(queries) < 3:
                queries.append(f"{query} 深度分析")

        print(f"[DeepSearch] 查询改写: {len(queries)} 个角度")
        for i, q in enumerate(queries):
            print(f"  {i+1}. {q}")

        return queries

    except Exception as e:
        print(f"[DeepSearch] 查询改写失败: {e}, 使用原始查询")
        return [query]


# ===================== 并行搜索 =====================

def _parallel_search(queries: List[str], tavily_client, max_results: int = 5) -> Dict[str, Optional[dict]]:
    """
    硬编码规则: 对所有改写查询执行并行搜索，不可跳过。

    Args:
        queries: 搜索查询词列表
        tavily_client: Tavily 客户端实例
        max_results: 每个查询返回的最大结果数

    Returns:
        {query: search_result_dict_or_None}
    """
    results: Dict[str, Optional[dict]] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for q in queries:
            future = executor.submit(
                tavily_client.search,
                query=q,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
            )
            futures[future] = q

        for future in as_completed(futures):
            q = futures[future]
            try:
                results[q] = future.result(timeout=30)
                count = len(results[q].get('results', [])) if results[q] else 0
                print(f"[DeepSearch] ✓ '{q[:40]}...' → {count} 条")
            except Exception as e:
                print(f"[DeepSearch] ✗ '{q[:40]}...' → 失败: {e}")
                results[q] = None

    return results


# ===================== 结果聚合 =====================

def _aggregate_results(
    original_query: str,
    rewritten_queries: List[str],
    search_results: Dict[str, Optional[dict]],
    max_total: int = 20,
) -> str:
    """
    去重聚合搜索结果，构建高密度上下文。

    去重策略: URL 精确去重
    排序策略: 按原始顺序保留，来自不同角度的结果交错排列

    Args:
        original_query: 用户原始查询
        rewritten_queries: 改写后的搜索词列表
        search_results: 各搜索词的返回结果
        max_total: 最大返回结果数

    Returns:
        格式化的高密度上下文字符串
    """
    seen_urls: set = set()
    all_items: list = []

    for q, result in search_results.items():
        if not result or not result.get('results'):
            continue
        for item in result['results']:
            url = item.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_items.append({
                    'url': url,
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'source_query': q,
                })

    # 构建高密度上下文
    parts = []
    parts.append("=" * 60)
    parts.append(f"多角度深度搜索结果报告")
    parts.append("=" * 60)
    parts.append(f"原始查询: {original_query}")
    parts.append(f"搜索角度数: {len(rewritten_queries)}")
    parts.append(f"去重后结果数: {len(all_items)}")
    parts.append("")

    parts.append("## 搜索角度")
    for i, q in enumerate(rewritten_queries):
        status = "✓" if search_results.get(q) else "✗"
        result_count = (
            len(search_results[q].get('results', []))
            if search_results.get(q) and search_results[q].get('results')
            else 0
        )
        parts.append(f"  {status} 角度{i + 1}: \"{q}\" ({result_count} 条)")

    parts.append("")
    parts.append(f"## 聚合结果 (共 {min(len(all_items), max_total)} 条, 去重 {len(all_items)} 条)")

    for i, item in enumerate(all_items[:max_total]):
        parts.append(f"\n### [{i + 1}] {item['title']}")
        parts.append(f"URL: {item['url']}")
        parts.append(f"来源角度: \"{item['source_query'][:60]}\"")
        parts.append(f"摘要: {item['content'][:400]}")

    if len(all_items) > max_total:
        parts.append(f"\n... 另有 {len(all_items) - max_total} 条结果已省略 ...")

    return '\n'.join(parts)


# ===================== 主工具 =====================

@tool
def deep_search(query: str) -> str:
    """
    多角度深度网络搜索工具 (Multi-Angle Deep Search)。

    核心机制:
      1. LLM 自动将查询改写为 3~5 个不同角度的搜索词
      2. 硬编码并行执行所有搜索 (不可跳过)
      3. URL 去重 + 结果聚合，输出高密度全局上下文

    适用场景:
      - 需要全面了解某个话题时 (而非单一信息来源)
      - 需要多角度、多立场的信息对比时
      - 需要降低信息源偏见、获取全局视角时

    Args:
        query: 用户原始查询问题

    Returns:
        聚合后的高密度搜索报告，包含多角度搜索结果和去重摘要
    """
    from agent.llm import model
    from tools.tavily_tools import tavily_client

    if not tavily_client:
        return "Error: Tavily 客户端未初始化，请检查 TAVILY_API_KEY 配置。"

    monitor.report_tool("深度搜索工具", {"query": query})

    # Step 1: 查询改写 (LLM 生成 3~5 个不同角度)
    rewritten_queries = _rewrite_query(query, model)

    # Step 2: 硬编码并行搜索 (强制执行所有角度)
    search_results = _parallel_search(rewritten_queries, tavily_client)

    # Step 3: 去重聚合构建高密度上下文
    context = _aggregate_results(query, rewritten_queries, search_results)

    monitor.report_tool("深度搜索完成", {
        "angles": len(rewritten_queries),
        "dedup_results": sum(
            1 for r in search_results.values()
            if r and r.get('results')
        ),
    })

    return context
