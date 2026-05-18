"""
DeepAgents Search 统一评测入口

一键运行全部三项评测:
  python -m evaluation.cli --all

单独运行:
  python -m evaluation.cli --router
  python -m evaluation.cli --rag
  python -m evaluation.cli --sql

快速模式 (少量样本试跑):
  python -m evaluation.cli --all --quick
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def _resolve_path(path: str) -> str:
    """解析路径，支持相对路径和默认值"""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / path)


# ===================== 各项评测入口 =====================

async def _run_router(args) -> Dict:
    """BERT 路由评测"""
    from evaluation.router_eval import run_router_evaluation

    data_path = _resolve_path(
        args.router_data or str(PROJECT_ROOT / "router" / "data" / "annotation_template.json")
    )
    if args.quick:
        print("[Router] 快速模式: 仅评测前 10 条")

    return run_router_evaluation(
        data_path=data_path,
        threshold=args.router_threshold,
        output_path=str(DATA_DIR / "router_eval_report.json"),
    )


async def _run_rag(args) -> Dict:
    """RAG 管线评测"""
    from rag_agent.scripts.evaluate import run_evaluation as rag_run

    testset_path = _resolve_path(
        args.rag_testset or str(DATA_DIR / "testset.json")
    )
    if not Path(testset_path).exists():
        return {"error": f"RAG 测试集不存在: {testset_path}\n请先运行: python -m rag_agent.scripts.evaluate --generate --size 30"}

    if args.quick:
        # 只跑前 5 条
        import json
        with open(testset_path, "r", encoding="utf-8") as f:
            full_testset = json.load(f)
        quick_path = str(DATA_DIR / "testset_quick.json")
        with open(quick_path, "w", encoding="utf-8") as f:
            json.dump(full_testset[:5], f, ensure_ascii=False, indent=2)
        testset_path = quick_path
        print("[RAG] 快速模式: 仅评测前 5 条")

    return rag_run(
        testset_path=testset_path,
        output_path=str(DATA_DIR / "rag_eval_report.json"),
        retrieval_only=args.rag_retrieval_only,
    )


async def _run_sql(args) -> Dict:
    """SQL Agent 评测"""
    from sql_agent.evaluation.evaluate import run_evaluation as sql_run

    testset_path = _resolve_path(
        args.sql_testset or str(DATA_DIR / "sql_testset.json")
    )
    if not Path(testset_path).exists():
        return {"error": f"SQL 测试集不存在: {testset_path}\n请先运行: python -m sql_agent.evaluation.testset_generator --size 200"}

    limit = args.quick and 10 or args.sql_limit or 0

    return await sql_run(
        testset_path=testset_path,
        output_path=str(DATA_DIR / "sql_eval_report.json"),
        limit=limit,
        concurrency=args.sql_concurrency or 5,
    )


# ===================== 主入口 =====================

async def run_all(args) -> Dict:
    """运行全部评测"""
    results = {}

    print("\n" + "=" * 60)
    print("  DeepAgents Search 统一评测")
    print("=" * 60)

    # 1. Router
    print("\n" + "-" * 40)
    print("  [1/3] BERT 路由评测")
    print("-" * 40)
    try:
        results["router"] = await _run_router(args)
    except Exception as e:
        print(f"  [Router] 评测失败: {e}")
        results["router"] = {"error": str(e)}

    # 2. RAG
    print("\n" + "-" * 40)
    print("  [2/3] RAG 管线评测")
    print("-" * 40)
    try:
        results["rag"] = await _run_rag(args)
    except Exception as e:
        print(f"  [RAG] 评测失败: {e}")
        results["rag"] = {"error": str(e)}

    # 3. SQL
    print("\n" + "-" * 40)
    print("  [3/3] SQL Agent 评测")
    print("-" * 40)
    try:
        results["sql"] = await _run_sql(args)
    except Exception as e:
        print(f"  [SQL] 评测失败: {e}")
        results["sql"] = {"error": str(e)}

    # 汇总
    _print_summary(results)
    _save_summary(results)

    return results


def _print_summary(results: Dict):
    """打印统一评测汇总"""
    print("\n" + "=" * 60)
    print("  评测汇总")
    print("=" * 60)

    # Router
    r = results.get("router", {})
    if r.get("error"):
        print(f"  BERT 路由: ✗ {r['error'][:80]}")
    else:
        em = r.get("exact_match", 0)
        bar = "█" * int(em * 20) + "░" * (20 - int(em * 20))
        print(f"  BERT 路由: Exact Match = {em:.2%} {bar}")

    # RAG
    rag = results.get("rag", {})
    if rag.get("error"):
        print(f"  RAG 管线:  ✗ {rag['error'][:80]}")
    else:
        scores = rag.get("scores", {})
        faith = scores.get("faithfulness", "N/A")
        relev = scores.get("answer_relevancy", "N/A")
        print(f"  RAG 管线:  Faithfulness={_fmt(faith)}  AnswerRelevancy={_fmt(relev)}")

    # SQL
    sql = results.get("sql", {})
    if sql.get("error"):
        print(f"  SQL Agent:  ✗ {sql['error'][:80]}")
    else:
        ex = sql.get("execution_accuracy", 0)
        bar = "█" * int(ex * 20) + "░" * (20 - int(ex * 20))
        print(f"  SQL Agent:  EX = {ex:.2%} {bar}")


def _fmt(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.2%}"
    return str(v)


def _save_summary(results: Dict):
    """保存汇总报告"""
    summary = {}
    for key in ["router", "rag", "sql"]:
        r = results.get(key, {})
        if r.get("error"):
            summary[key] = {"status": "error", "message": r["error"]}
        elif key == "router":
            summary[key] = {
                "status": "ok",
                "exact_match": r.get("exact_match"),
                "macro_f1": r.get("macro_f1"),
                "per_label": r.get("per_label"),
            }
        elif key == "rag":
            summary[key] = {
                "status": "ok",
                "scores": r.get("scores"),
            }
        elif key == "sql":
            summary[key] = {
                "status": "ok",
                "execution_accuracy": r.get("execution_accuracy"),
                "total_cases": r.get("total_cases"),
                "error_distribution": r.get("error_distribution"),
            }

    path = str(DATA_DIR / "eval_summary.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总已保存: {path}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="DeepAgents Search 统一评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m evaluation.cli --all              # 全量评测
  python -m evaluation.cli --all --quick       # 快速模式 (少量样本)
  python -m evaluation.cli --router            # 仅路由
  python -m evaluation.cli --rag               # 仅 RAG
  python -m evaluation.cli --sql               # 仅 SQL
  python -m evaluation.cli --sql --sql-limit 20  # SQL 评测限 20 条
        """,
    )

    # 评测选择
    parser.add_argument("--all", action="store_true", help="运行全部三项评测")
    parser.add_argument("--router", action="store_true", help="BERT 路由评测")
    parser.add_argument("--rag", action="store_true", help="RAG 管线评测")
    parser.add_argument("--sql", action="store_true", help="SQL Agent 评测")
    parser.add_argument("--quick", action="store_true", help="快速模式 (少量样本)")

    # Router 参数
    parser.add_argument("--router-data", type=str, default=None,
                        help="路由标注数据路径")
    parser.add_argument("--router-threshold", type=float, default=None,
                        help="多标签阈值")

    # RAG 参数
    parser.add_argument("--rag-testset", type=str, default=None,
                        help="RAG 测试集路径")
    parser.add_argument("--rag-retrieval-only", action="store_true",
                        help="仅评测检索层")

    # SQL 参数
    parser.add_argument("--sql-testset", type=str, default=None,
                        help="SQL 测试集路径")
    parser.add_argument("--sql-limit", type=int, default=0,
                        help="SQL 评测条数限制 (0=全部)")
    parser.add_argument("--sql-concurrency", type=int, default=5,
                        help="SQL 并发数")

    args = parser.parse_args()

    # 默认全跑
    if not any([args.router, args.rag, args.sql, args.all]):
        args.all = True

    if args.all:
        asyncio.run(run_all(args))
        return

    # 单项
    if args.router:
        asyncio.run(_run_router(args))
    if args.rag:
        asyncio.run(_run_rag(args))
    if args.sql:
        asyncio.run(_run_sql(args))


if __name__ == "__main__":
    main()
