"""
RAG 管线评测脚本

支持两种模式:
  1. 生成测试集: LLM 从已有文档自动生成问题 + 标准答案
  2. 运行评测: 跑检索管线 → RAGAS 自动打分

依赖:
  pip install ragas

使用方式:
  # 第一步: 生成测试集（LLM 自动出题，人工校验后保存）
  python -m rag_agent.scripts.evaluate --generate --size 30

  # 第二步: 运行评测
  python -m rag_agent.scripts.evaluate --evaluate --testset data/testset.json

  # 一键全流程
  python -m rag_agent.scripts.evaluate --all --size 30

  # 只测检索层（不跑 LLM 生成，省 token）
  python -m rag_agent.scripts.evaluate --evaluate --testset data/testset.json --retrieval_only

输出:
  终端打印评测报告（含每指标均值 + 最差问题 TOP5）
  同时保存 JSON 到 data/eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================== 测试集生成 =====================

QUESTION_GENERATION_PROMPT = """根据以下文档内容，生成 3 个用户可能会问的问题及其标准答案。

要求:
  - 第 1 题: 简单事实型（单段文档可直接回答）
  - 第 2 题: 需要推理或综合（跨段落或多个事实）
  - 第 3 题: 跨文档综合型（需要结合多篇文档才能回答）

输出格式 (严格 JSON，不要其他文字):
[
  {{"question": "...", "ground_truth": "...", "type": "simple"}},
  {{"question": "...", "ground_truth": "...", "type": "reasoning"}},
  {{"question": "...", "ground_truth": "...", "type": "multi_context"}}
]

文档内容:
{doc_content}"""


def load_documents(documents_dir: str = None) -> List[Dict]:
    """从文档目录加载所有文档"""
    from rag_agent.loader import DocumentLoader

    if documents_dir is None:
        documents_dir = str(PROJECT_ROOT / "data" / "documents")

    loader = DocumentLoader(documents_dir)
    return loader.load()


def generate_testset(
    num_questions: int = 30,
    documents_dir: str = None,
    output_path: str = None,
    llm=None,
) -> List[Dict]:
    """
    LLM 自动生成测试集。

    策略:
      每篇文档生成 3 个问题（简单/推理/综合），
      不够 num_questions 就多轮采样，超过就截断。
    """
    documents = load_documents(documents_dir)

    if not documents:
        print("[Eval] 错误: data/documents/ 目录下没有文档，请先放入文档")
        return []

    if llm is None:
        from agent.llm import model as default_llm
        llm = default_llm

    num_docs = min(len(documents), max(3, num_questions // 3))
    sampled_docs = np.random.choice(documents, size=num_docs, replace=False)

    all_questions = []
    for i, doc in enumerate(sampled_docs):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        content = content[:2000]  # 截断，避免超 token

        prompt = QUESTION_GENERATION_PROMPT.format(doc_content=content)

        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)

            # 提取 JSON（可能包裹在 markdown 代码块里）
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            questions = json.loads(text)
            for q in questions:
                q["source_doc"] = getattr(doc, 'metadata', {}).get('source', f"doc_{i}")
            all_questions.extend(questions)
            print(f"  ✓ 文档 {i+1}/{num_docs}: 生成 {len(questions)} 题")

        except Exception as e:
            print(f"  ✗ 文档 {i+1}/{num_docs}: 生成失败 - {e}")

    # 截断到目标数量，保证题型均衡
    if len(all_questions) > num_questions:
        types = {"simple": [], "reasoning": [], "multi_context": []}
        for q in all_questions:
            t = q.get("type", "simple")
            types.setdefault(t, []).append(q)

        balanced = []
        per_type = num_questions // 3
        for t in ["simple", "reasoning", "multi_context"]:
            balanced.extend(types.get(t, [])[:per_type])

        # 不够的用剩余补充
        remaining = num_questions - len(balanced)
        if remaining > 0:
            extras = [q for q in all_questions if q not in balanced][:remaining]
            balanced.extend(extras)

        all_questions = balanced[:num_questions]

    # 保存
    output_path = output_path or str(PROJECT_ROOT / "data" / "testset.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"\n[Eval] 测试集已保存: {output_path} ({len(all_questions)} 题)")
    print("[Eval] 建议人工检查后运行 --evaluate")

    return all_questions


# ===================== 评测执行 =====================

def run_evaluation(
    testset_path: str,
    output_path: str = None,
    retrieval_only: bool = False,
) -> Dict:
    """
    使用 RAGAS 对 RAG 管线进行评测。

    Args:
        testset_path: 测试集 JSON 路径
        output_path: 报告输出路径
        retrieval_only: 仅评测检索层（省 LLM token）

    Returns:
        评测指标字典
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    from rag_agent.retriever import get_retriever

    # 1. 加载测试集
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    if not testset:
        print("[Eval] 测试集为空")
        return {}

    print(f"[Eval] 加载测试集: {len(testset)} 题")

    # 2. 对每个问题跑检索管线
    retriever = get_retriever()
    eval_data = {
        "question": [],
        "ground_truth": [],
        "contexts": [],
    }
    if not retrieval_only:
        eval_data["answer"] = []

    from agent.llm import model as llm

    for i, item in enumerate(testset):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        # 检索
        try:
            results = retriever.search(question)
            contexts = [r.get("content", "") for r in results]
        except Exception as e:
            print(f"  ✗ 检索失败 [{i+1}]: {e}")
            contexts = []

        eval_data["question"].append(question)
        eval_data["ground_truth"].append(ground_truth)
        eval_data["contexts"].append(contexts)

        # 生成答案（非检索层专用模式）
        if not retrieval_only:
            try:
                context_text = "\n\n---\n\n".join(contexts[:5])
                gen_prompt = (
                    f"根据以下参考信息回答问题。\n\n"
                    f"参考信息:\n{context_text}\n\n"
                    f"问题: {question}\n\n"
                    f"回答:"
                )
                response = llm.invoke(gen_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"  ✗ 生成失败 [{i+1}]: {e}")
                answer = ""

            eval_data["answer"].append(answer)

        print(f"  [{i+1}/{len(testset)}] {question[:60]}...")

    # 3. RAGAS 评分
    dataset = Dataset.from_dict(eval_data)

    if retrieval_only:
        metrics = [context_precision, context_recall]
        metric_names = ["context_precision", "context_recall"]
    else:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"\n[Eval] RAGAS 评分中 (metrics={metric_names})...")
    result = evaluate(dataset, metrics=metrics)
    scores = {name: float(result[name]) for name in metric_names}

    # 4. 找最差问题
    per_question_scores = {}
    for name in metric_names:
        if hasattr(result, f"{name}_per_sample"):
            per_question_scores[name] = [
                float(s) if s is not None else None
                for s in getattr(result, f"{name}_per_sample")
            ]

    worst = _find_worst(testset, per_question_scores, top_n=5)

    # 5. 输出报告
    report = {
        "testset_path": testset_path,
        "num_questions": len(testset),
        "retrieval_only": retrieval_only,
        "scores": scores,
        "worst_questions": worst,
    }

    _print_report(report)

    # 6. 保存
    output_path = output_path or str(PROJECT_ROOT / "data" / "eval_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] 报告已保存: {output_path}")

    return report


def _find_worst(
    testset: List[Dict],
    per_question_scores: Dict[str, List[Optional[float]]],
    top_n: int = 5,
) -> List[Dict]:
    """找出得分最低的问题（用于定位管线问题）"""
    # 综合所有指标求平均
    all_scores = []
    for i, _ in enumerate(testset):
        item_scores = []
        for name, scores in per_question_scores.items():
            if i < len(scores) and scores[i] is not None:
                item_scores.append(scores[i])
        avg = np.mean(item_scores) if item_scores else 0.0
        all_scores.append({"index": i, "avg_score": avg})

    all_scores.sort(key=lambda x: x["avg_score"])

    worst = []
    for entry in all_scores[:top_n]:
        i = entry["index"]
        q = testset[i]
        detail = {"question": q["question"][:120], "avg_score": round(entry["avg_score"], 3)}
        for name, scores in per_question_scores.items():
            if i < len(scores) and scores[i] is not None:
                detail[name] = round(scores[i], 3)
        worst.append(detail)

    return worst


def _print_report(report: Dict):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("  RAG 评测报告")
    print("=" * 60)
    print(f"  测试题数: {report['num_questions']}")
    print(f"  模式: {'仅检索层' if report['retrieval_only'] else '端到端'}")
    print()

    for name, score in report["scores"].items():
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        status = "✓" if score >= 0.7 else ("⚠" if score >= 0.5 else "✗")
        print(f"  {status} {name:<22s}: {score:.3f}  {bar}")

    if report.get("worst_questions"):
        print(f"\n  得分最低的 {len(report['worst_questions'])} 题:")
        for i, wq in enumerate(report["worst_questions"], 1):
            print(f"  {i}. [{wq['avg_score']:.2f}] {wq['question']}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="RAG 管线评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m rag_agent.scripts.evaluate --generate --size 30
  python -m rag_agent.scripts.evaluate --evaluate --testset data/testset.json
  python -m rag_agent.scripts.evaluate --evaluate --testset data/testset.json --retrieval_only
  python -m rag_agent.scripts.evaluate --all --size 30
        """,
    )

    parser.add_argument("--generate", action="store_true", help="生成测试集")
    parser.add_argument("--evaluate", action="store_true", help="运行评测")
    parser.add_argument("--all", action="store_true", help="生成 + 评测一键全流程")
    parser.add_argument("--size", type=int, default=30, help="测试集题目数量 (默认 30)")
    parser.add_argument("--testset", type=str, default=None, help="测试集 JSON 路径")
    parser.add_argument("--retrieval_only", action="store_true",
                        help="仅评测检索层（不跑 LLM 生成，省 token）")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")

    args = parser.parse_args()

    if args.all:
        args.generate = True
        args.evaluate = True

    testset_path = args.testset or str(PROJECT_ROOT / "data" / "testset.json")
    report_path = args.output or str(PROJECT_ROOT / "data" / "eval_report.json")

    # Step 1: 生成测试集
    if args.generate:
        print("\n[Step 1] 生成测试集...")
        testset = generate_testset(
            num_questions=args.size,
            output_path=testset_path,
        )
        if not testset:
            return
        print("  请检查生成的测试集，确认后运行 --evaluate")

    # Step 2: 评测
    if args.evaluate:
        if not Path(testset_path).exists():
            print(f"[Eval] 测试集不存在: {testset_path}")
            print("  请先运行 --generate 生成测试集")
            return

        print("\n[Step 2] 运行评测...")
        run_evaluation(
            testset_path=testset_path,
            output_path=report_path,
            retrieval_only=args.retrieval_only,
        )


if __name__ == "__main__":
    main()
