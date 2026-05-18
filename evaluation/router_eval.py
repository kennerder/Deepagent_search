"""
BERT 路由评测

评测指标:
  - Accuracy: 预测标签集合与真实标签集合完全一致的比例

数据格式 (与 annotation_template.json 一致):
  [
    {"query": "...", "labels": ["db_query", "web_search"]},
    {"query": "...", "labels": ["web_search"]},
    ...
  ]

使用:
  python -m evaluation.router_eval --data router/data/annotation_template.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from router.config import SUB_AGENT_LABELS, LABEL_CN_MAP


# ===================== 数据加载 =====================

def load_test_data(data_path: str) -> List[Dict]:
    """加载测试数据，统一为多标签格式"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = []
    for item in data:
        if "labels" in item:
            labels = item["labels"]
        elif "label" in item:
            labels = [item["label"]]
        else:
            continue
        normalized.append({"query": item["query"], "labels": labels})

    return normalized


# ===================== 评测执行 =====================

def run_router_evaluation(
    data_path: str,
    threshold: float = None,
    output_path: str = None,
) -> Dict:
    """
    执行 BERT 路由评测。

    Args:
        data_path: 标注数据路径
        threshold: 多标签阈值 (默认使用配置值)
        output_path: 报告输出路径

    Returns:
        {"total": int, "correct": int, "accuracy": float, "wrong_cases": [...]}
    """
    from router.classifier import get_classifier
    from router.config import router_config

    if threshold is not None:
        router_config.multi_label_threshold = threshold

    # 1. 加载数据
    data = load_test_data(data_path)
    if not data:
        print("[RouterEval] 没有可用数据")
        return {}

    # 统计
    single_count = sum(1 for d in data if len(d["labels"]) == 1)
    multi_count = sum(1 for d in data if len(d["labels"]) > 1)
    label_counts = {l: sum(l in d["labels"] for d in data) for l in SUB_AGENT_LABELS}

    print(f"[RouterEval] 数据: {len(data)} 条 "
          f"(单标签: {single_count}, 多标签: {multi_count})")
    for label, count in label_counts.items():
        print(f"  {LABEL_CN_MAP.get(label, label)}: {count}")

    # 2. 加载模型
    classifier = get_classifier()
    try:
        classifier.load()
    except Exception as e:
        print(f"[RouterEval] 模型加载失败: {e}")
        return {"error": f"模型加载失败: {e}"}

    # 3. 逐条预测，统计准确率
    total = 0
    correct = 0
    wrong_cases = []

    for item in data:
        query = item["query"]
        true_labels = set(item["labels"])

        try:
            result = classifier.predict(query)
            pred_labels = set(result.get("labels", []))
        except Exception as e:
            pred_labels = set()

        total += 1
        if true_labels == pred_labels:
            correct += 1
        else:
            wrong_cases.append({
                "query": query[:120],
                "expected": sorted(true_labels),
                "predicted": sorted(pred_labels),
                "missed": sorted(true_labels - pred_labels),
                "false_positive": sorted(pred_labels - true_labels),
            })

    accuracy = correct / total if total > 0 else 0

    report = {
        "data_path": data_path,
        "threshold": router_config.multi_label_threshold,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "wrong_cases": wrong_cases,
    }

    _print_report(report)

    # 4. 保存
    output_path = output_path or str(PROJECT_ROOT / "data" / "router_eval_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[RouterEval] 报告已保存: {output_path}")

    return report


def _print_report(report: Dict):
    """打印路由评测报告"""
    print("\n" + "=" * 60)
    print("  BERT 路由评测报告")
    print("=" * 60)
    print(f"  样本数: {report['total']}")
    print(f"  正确: {report['correct']}  错误: {report['total'] - report['correct']}")
    bar = "█" * int(report['accuracy'] * 30) + "░" * (30 - int(report['accuracy'] * 30))
    print(f"  准确率: {report['accuracy']:.2%}  {bar}")

    if report.get("wrong_cases"):
        print(f"\n  ▸ 错误案例 ({min(10, len(report['wrong_cases']))} 条样例):")
        for wc in report["wrong_cases"][:10]:
            missed = wc.get("missed", [])
            fp = wc.get("false_positive", [])
            details = []
            if missed:
                details.append(f"漏了 {missed}")
            if fp:
                details.append(f"多判了 {fp}")
            print(f"    Q: {wc['query']}")
            print(f"    期望: {wc['expected']}  预测: {wc['predicted']}  {'/'.join(details)}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(description="BERT 路由评测")
    parser.add_argument("--data", type=str, default=None,
                        help="标注数据路径 (默认 router/data/annotation_template.json)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="多标签阈值 (默认 0.5)")
    parser.add_argument("--output", type=str, default=None,
                        help="报告输出路径")
    args = parser.parse_args()

    data_path = args.data or str(PROJECT_ROOT / "router" / "data" / "annotation_template.json")

    if not Path(data_path).exists():
        print(f"数据文件不存在: {data_path}")
        return

    run_router_evaluation(
        data_path=data_path,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
