"""
SQL Agent 评测脚本

基于 BIRD-SQL 评测范式，测试 SQL 语句执行准确率 (EX)。

评测指标:
  - EX (Execution Accuracy): 生成 SQL 的执行结果 == 标准 SQL 的执行结果
  - 按难度/类别分组的准确率
  - 错误分类: 语法错 / 语义错 / 空结果 / 超时

使用:
  python -m sql_agent.evaluation.evaluate --testset data/sql_testset.json
  python -m sql_agent.evaluation.evaluate --testset data/sql_testset.json --limit 10  # 快速抽样
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================== 结果比对 =====================

def _normalize_result(raw: str) -> str:
    """标准化执行结果字符串，消除格式差异"""
    return raw.strip().replace(" ", "").replace("\n", "").lower()


def _compare_results(result_a: str, result_b: str) -> bool:
    """
    比对两个 SQL 执行结果。

    策略: 标准化后字符串比对。
    对排序敏感的查询可能误判，但能覆盖绝大多数情况。
    """
    return _normalize_result(str(result_a)) == _normalize_result(str(result_b))


def _classify_error(
    generated_sql: str,
    generated_result: str,
    expected_sql: str,
    expected_result: str,
    error_msg: str = "",
) -> str:
    """
    分类 SQL 错误类型。

    Returns:
        "语法错误" | "语义错误(表/列召回错误)" | "空结果" | "结果不匹配" | "超时" | "未知"
    """
    if error_msg:
        if "timeout" in error_msg.lower():
            return "超时"
        if any(kw in error_msg.lower() for kw in ["syntax", "error", "exist", "unknown column", "table"]):
            return "语法错误"
        return "未知"

    if generated_result is None or "error" in str(generated_result).lower():
        return "语法错误"

    gen_str = str(generated_result).strip()
    exp_str = str(expected_result).strip()

    if gen_str in ("[]", "()", "", "None", "null"):
        if exp_str not in ("[]", "()", "", "None", "null"):
            return "空结果"

    # 检查是否可能是表/列选错
    if generated_sql and expected_sql:
        gen_tables = _extract_table_names(generated_sql)
        exp_tables = _extract_table_names(expected_sql)
        if gen_tables != exp_tables:
            return "语义错误(表/列召回错误)"

    return "结果不匹配"


def _extract_table_names(sql: str) -> set:
    """从 SQL 中提取表名"""
    import re
    sql_upper = sql.upper()
    tables = set()
    for match in re.finditer(r'(?:FROM|JOIN)\s+(\w+)', sql_upper):
        tables.add(match.group(1).lower())
    return tables


# ===================== 单条评测 =====================

async def _evaluate_one(
    case: Dict,
    dw_repo,
    graph,
) -> Dict:
    """
    对单条测试用例执行评测。

    Returns:
        {
            "id": ..., "question": ..., "pass": bool,
            "generated_sql": ..., "expected_sql": ...,
            "error_type": ... (if not pass),
            "elapsed_ms": ...
        }
    """
    question = case["question"]
    expected_sql = case["expected_sql"]
    start = time.time()

    result = {
        "id": case["id"],
        "question": question,
        "difficulty": case.get("difficulty", ""),
        "category": case.get("category", ""),
        "expected_sql": expected_sql,
    }

    try:
        # 1. 执行标准 SQL，获取标准结果
        try:
            expected_result = await dw_repo.run_sql(expected_sql)
            result["expected_result"] = str(expected_result)
        except Exception as e:
            result["expected_result"] = f"[标准SQL执行失败] {e}"
            # 标准 SQL 都失败了，这条题有问题，跳过
            result["pass"] = None
            result["error_type"] = "标准SQL无效"
            result["elapsed_ms"] = int((time.time() - start) * 1000)
            return result

        # 2. 跑 LangGraph 流水线
        initial_state = {
            "query": question,
            "keywords": [],
            "retrieved_column_infos": [],
            "retrieved_metric_infos": [],
            "retrieved_value_infos": [],
            "error": None,
            "table_infos": [],
            "metric_infos": [],
            "sql": "",
            "date_info": {"date": "", "weekday": "", "quarter": ""},
            "db_info": {"dialect": "mysql", "version": "8.0"},
            "messages": [],
        }

        # 跑图，设置超时
        final_state = None
        try:
            async for chunk in graph.astream(
                initial_state,
                config={"recursion_limit": 30},
            ):
                # chunk 是 {node_name: state_update}
                for node_name, state_update in chunk.items():
                    if state_update:
                        final_state = state_update
        except asyncio.TimeoutError:
            result["pass"] = False
            result["error_type"] = "超时"
            result["elapsed_ms"] = int((time.time() - start) * 1000)
            return result

        # 3. 提取生成的 SQL
        generated_sql = ""
        if final_state:
            generated_sql = final_state.get("sql", "")

        # 如果最终状态没有 sql，从 messages 里提取
        if not generated_sql and final_state and "messages" in final_state:
            for msg in reversed(final_state["messages"]):
                content = str(msg.get("content", ""))
                if "sql" in content.lower() or "SELECT" in content.upper():
                    # 尝试从输出中解析
                    import re
                    match = re.search(r"'sql'\s*:\s*'([^']+)'", content)
                    if match:
                        generated_sql = match.group(1)
                    break

        result["generated_sql"] = generated_sql or ""

        # 4. 执行生成的 SQL
        if generated_sql:
            try:
                generated_result = await dw_repo.run_sql(generated_sql)
                result["generated_result"] = str(generated_result)
            except Exception as e:
                result["generated_result"] = f"[执行失败] {e}"
                result["pass"] = False
                result["error_type"] = _classify_error(
                    generated_sql, None, expected_sql, str(expected_result), str(e)
                )
                result["elapsed_ms"] = int((time.time() - start) * 1000)
                return result
        else:
            result["generated_result"] = "[未生成SQL]"
            result["pass"] = False
            result["error_type"] = "语法错误"
            result["elapsed_ms"] = int((time.time() - start) * 1000)
            return result

        # 5. 比对结果
        passed = _compare_results(
            result.get("generated_result", ""),
            str(expected_result),
        )
        result["pass"] = passed
        if not passed:
            result["error_type"] = _classify_error(
                generated_sql,
                result.get("generated_result", ""),
                expected_sql,
                str(expected_result),
            )

        result["elapsed_ms"] = int((time.time() - start) * 1000)

    except Exception as e:
        result["pass"] = False
        result["error_type"] = f"评测异常: {e}"
        result["elapsed_ms"] = int((time.time() - start) * 1000)

    return result


# ===================== 批量评测 =====================

async def run_evaluation(
    testset_path: str,
    output_path: str = None,
    limit: int = 0,
    concurrency: int = 5,
) -> Dict:
    """
    批量执行 SQL Agent 评测。

    Args:
        testset_path: 测试集 JSON 路径
        output_path: 报告输出路径
        limit: 限制测试条数 (0 = 全部)
        concurrency: 并发数 (数据库连接限制)

    Returns:
        评测报告字典
    """
    # 延迟导入，避免循环依赖
    from sql_agent.clients.mysql_client_manager import mysql_client_manager
    from sql_agent.agent.graph import sql_graph

    # 1. 加载测试集
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    if limit > 0:
        testset = testset[:limit]

    print(f"[Eval] 加载测试集: {len(testset)} 条")

    # 2. 初始化数据库连接
    print("[Eval] 初始化数据库连接...")
    mysql_client_manager.init()
    dw_repo = mysql_client_manager.dw_repository

    # 3. 逐条评测 (控制并发)
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_with_limit(case):
        async with semaphore:
            return await _evaluate_one(case, dw_repo, sql_graph)

    print(f"[Eval] 开始评测 (并发={concurrency})...")
    tasks = [evaluate_with_limit(case) for case in testset]
    results = await asyncio.gather(*tasks)

    # 4. 汇总统计
    valid_results = [r for r in results if r.get("pass") is not None]
    total = len(valid_results)
    passed = sum(1 for r in valid_results if r["pass"])

    # 按难度分组
    by_difficulty = {}
    for r in valid_results:
        d = r.get("difficulty", "unknown")
        if d not in by_difficulty:
            by_difficulty[d] = {"total": 0, "passed": 0}
        by_difficulty[d]["total"] += 1
        if r["pass"]:
            by_difficulty[d]["passed"] += 1

    # 按类别分组
    by_category = {}
    for r in valid_results:
        c = r.get("category", "unknown")
        if c not in by_category:
            by_category[c] = {"total": 0, "passed": 0}
        by_category[c]["total"] += 1
        if r["pass"]:
            by_category[c]["passed"] += 1

    # 错误分类
    error_counts = {}
    for r in valid_results:
        if not r["pass"]:
            etype = r.get("error_type", "未知")
            error_counts[etype] = error_counts.get(etype, 0) + 1

    # 平均耗时
    elapsed_times = [r.get("elapsed_ms", 0) for r in valid_results]
    avg_elapsed = int(np.mean(elapsed_times)) if elapsed_times else 0

    report = {
        "testset_path": testset_path,
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "execution_accuracy": round(passed / total, 4) if total > 0 else 0,
        "avg_elapsed_ms": avg_elapsed,
        "by_difficulty": {
            d: {
                "accuracy": round(v["passed"] / v["total"], 4) if v["total"] > 0 else 0,
                "total": v["total"],
            }
            for d, v in sorted(by_difficulty.items())
        },
        "by_category": {
            c: {
                "accuracy": round(v["passed"] / v["total"], 4) if v["total"] > 0 else 0,
                "total": v["total"],
            }
            for c, v in sorted(by_category.items())
        },
        "error_distribution": dict(sorted(error_counts.items(), key=lambda x: -x[1])),
        "failed_cases": [
            {
                "id": r["id"],
                "question": r["question"][:100],
                "error_type": r.get("error_type", ""),
                "generated_sql": r.get("generated_sql", "")[:200],
                "expected_sql": r.get("expected_sql", "")[:200],
            }
            for r in valid_results if not r["pass"]
        ][:20],
    }

    _print_report(report)

    # 5. 保存
    output_path = output_path or str(PROJECT_ROOT / "data" / "sql_eval_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] 报告已保存: {output_path}")

    return report


def _print_report(report: Dict):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("  SQL Agent 评测报告 (BIRD-style EX)")
    print("=" * 60)
    print(f"  测试题数: {report['total_cases']}")
    print(f"  通过: {report['passed']}  失败: {report['failed']}")
    print(f"  执行准确率 (EX): {report['execution_accuracy']:.2%}")
    print(f"  平均耗时: {report['avg_elapsed_ms']}ms")
    print()

    print("  ▸ 按难度:")
    for d, v in report["by_difficulty"].items():
        bar = "█" * int(v["accuracy"] * 20) + "░" * (20 - int(v["accuracy"] * 20))
        print(f"    {d:<10s}: {v['accuracy']:.0%} {bar} ({v['total']}题)")

    print("\n  ▸ 按类别:")
    for c, v in report["by_category"].items():
        bar = "█" * int(v["accuracy"] * 20) + "░" * (20 - int(v["accuracy"] * 20))
        print(f"    {c:<12s}: {v['accuracy']:.0%} {bar} ({v['total']}题)")

    if report["error_distribution"]:
        print("\n  ▸ 错误分布:")
        for etype, count in report["error_distribution"].items():
            print(f"    {etype}: {count} 条")

    if report["failed_cases"]:
        print(f"\n  ▸ 失败案例 ({min(5, len(report['failed_cases']))} 条样例):")
        for fc in report["failed_cases"][:5]:
            print(f"    [{fc['id']}] {fc['error_type']}")
            print(f"    Q: {fc['question']}")
            print(f"    Gen: {fc['generated_sql'][:120]}")
            print(f"    Exp: {fc['expected_sql'][:120]}")
            print()


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="SQL Agent 评测 (BIRD-style EX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m sql_agent.evaluation.evaluate --testset data/sql_testset.json
  python -m sql_agent.evaluation.evaluate --testset data/sql_testset.json --limit 10
  python -m sql_agent.evaluation.evaluate --testset data/sql_testset.json --concurrency 3
        """,
    )
    parser.add_argument("--testset", type=str, required=True, help="测试集 JSON 路径")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    parser.add_argument("--limit", type=int, default=0, help="限制测试条数 (0=全部)")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数 (默认 5)")
    args = parser.parse_args()

    if not Path(args.testset).exists():
        print(f"测试集不存在: {args.testset}")
        print("请先运行: python -m sql_agent.evaluation.testset_generator")
        return

    asyncio.run(run_evaluation(
        testset_path=args.testset,
        output_path=args.output,
        limit=args.limit,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
