"""
BERT 路由模型微调训练器 (多标签模式)

训练流程:
  1. 加载标注数据 (JSON 格式，支持单标签和多标签)
  2. 数据划分 (train/val split)
  3. hfl/chinese-roberta-wwm-ext + 多标签分类头 微调
  4. 评估 + 保存

标注数据格式 (router/data/annotations.json):
  单标签 (兼容旧格式):
    {"query": "查询布洛芬的库存数量", "label": "db_query"}

  多标签 (推荐):
    {"query": "查布洛芬库存并搜索最新市场行情", "labels": ["db_query", "web_search"]}

多标签设计:
  使用 sigmoid + BCEWithLogitsLoss，每个标签独立判断，允许同时激活多个子代理。
  这解决了"查库存+搜行情"这类跨子代理查询的路由问题。

使用方式:
  python -m router.trainer
  python -m router.trainer --data path/to/annotations.json
  python -m router.trainer --output models/my_router
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from router.config import router_config, SUB_AGENT_LABELS


class BertRouterTrainer:
    """
    BERT 路由模型训练器 (多标签模式)

    基于 hfl/chinese-roberta-wwm-ext 做多标签序列分类微调。
    使用 sigmoid + BCEWithLogitsLoss，每个子代理标签独立判断。
    """

    def __init__(
        self,
        annotations_path: str = None,
        output_dir: str = None,
    ):
        self.annotations_path = annotations_path or str(
            Path(__file__).parent / "data" / "annotations.json"
        )
        self.output_dir = output_dir or router_config.finetuned_model_dir
        self._label_to_id = {label: i for i, label in enumerate(SUB_AGENT_LABELS)}
        self._id_to_label = {i: label for i, label in enumerate(SUB_AGENT_LABELS)}
        self._num_labels = len(SUB_AGENT_LABELS)

    # ===================== 数据加载 =====================

    def load_annotations(self) -> List[Dict]:
        """
        加载标注数据。支持单标签和多标签两种格式:

          单标签: {"query": "...", "label": "db_query"}
          多标签: {"query": "...", "labels": ["db_query", "web_search"]}

        Returns:
            统一为 [{"query": str, "labels": [str, ...]}, ...]
        """
        path = Path(self.annotations_path)
        if not path.exists():
            raise FileNotFoundError(
                f"标注数据文件不存在: {path}\n"
                f"请先创建标注数据，参考 router/data/annotation_template.json"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        normalized = []
        for item in data:
            if "query" not in item:
                raise ValueError(f"数据格式错误，缺少 query: {item}")

            # 统一为多标签格式
            if "labels" in item:
                labels = item["labels"]
            elif "label" in item:
                labels = [item["label"]]
            else:
                raise ValueError(f"数据格式错误，缺少 label 或 labels: {item}")

            # 校验标签
            for label in labels:
                if label not in SUB_AGENT_LABELS:
                    raise ValueError(
                        f"未知标签 '{label}'，合法标签: {SUB_AGENT_LABELS}"
                    )

            normalized.append({"query": item["query"], "labels": labels})

        # 统计
        single_count = sum(1 for item in normalized if len(item["labels"]) == 1)
        multi_count = sum(1 for item in normalized if len(item["labels"]) > 1)
        print(f"[Trainer] 加载标注数据: {len(normalized)} 条 "
              f"(单标签: {single_count}, 多标签: {multi_count})")
        return normalized

    def _labels_to_multihot(self, labels_list: List[List[str]]) -> np.ndarray:
        """将标签列表转为 multi-hot 向量"""
        multi_hot = np.zeros((len(labels_list), self._num_labels), dtype=np.float32)
        for i, labels in enumerate(labels_list):
            for label in labels:
                multi_hot[i, self._label_to_id[label]] = 1.0
        return multi_hot

    def split_data(
        self, data: List[Dict], val_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict]]:
        """划分训练集和验证集 (分层抽样，尽量保留多标签样本在训练集)"""
        np.random.seed(42)

        # 分离单标签和多标签样本
        single = [d for d in data if len(d["labels"]) == 1]
        multi = [d for d in data if len(d["labels"]) > 1]

        # 多标签样本优先分入训练集 (因为更少更珍贵)
        np.random.shuffle(multi)
        multi_val_size = max(1, int(len(multi) * val_ratio))
        multi_val = multi[:multi_val_size]
        multi_train = multi[multi_val_size:]

        # 单标签样本按比例分
        np.random.shuffle(single)
        val_size = int(len(single) * val_ratio)
        single_val = single[:val_size]
        single_train = single[val_size:]

        train_data = single_train + multi_train
        val_data = single_val + multi_val
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)

        print(f"[Trainer] 训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")
        return train_data, val_data

    # ===================== 训练 =====================

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        val_ratio: float = 0.15,
        save_best: bool = True,
    ) -> Dict:
        """
        执行多标签微调训练。

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            val_ratio: 验证集比例
            save_best: 是否仅保存最佳模型

        Returns:
            训练指标字典
        """
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        import torch
        from datasets import Dataset

        # 1. 加载数据
        data = self.load_annotations()
        train_data, val_data = self.split_data(data, val_ratio)

        # 2. 加载分词器和模型 (多标签分类)
        tokenizer = AutoTokenizer.from_pretrained(router_config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            router_config.model_name,
            num_labels=router_config.num_labels,
            problem_type="multi_label_classification",
        )
        model.to(router_config.device)

        # 3. 编码数据 + multi-hot 标签
        train_labels = [d["labels"] for d in train_data]
        val_labels = [d["labels"] for d in val_data]
        train_multihot = self._labels_to_multihot(train_labels)
        val_multihot = self._labels_to_multihot(val_labels)

        def preprocess(examples):
            return tokenizer(
                examples["query"],
                max_length=router_config.max_length,
                padding="max_length",
                truncation=True,
            )

        train_dataset = Dataset.from_list(
            [{"query": d["query"]} for d in train_data]
        )
        train_dataset = train_dataset.map(preprocess, batched=True)
        train_dataset = train_dataset.add_column(
            "label", train_multihot.tolist()
        )

        val_dataset = Dataset.from_list(
            [{"query": d["query"]} for d in val_data]
        )
        val_dataset = val_dataset.map(preprocess, batched=True)
        val_dataset = val_dataset.add_column(
            "label", val_multihot.tolist()
        )

        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        # 4. 训练配置
        training_args = TrainingArguments(
            output_dir=str(Path(self.output_dir) / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=str(Path(self.output_dir) / "logs"),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=save_best,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=(router_config.device != "cpu"),
            dataloader_pin_memory=False,
            report_to="none",
        )

        # 5. 多标签评估指标
        threshold = router_config.multi_label_threshold
        num_labels = self._num_labels

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # sigmoid + threshold
            probs = 1.0 / (1.0 + np.exp(-logits))
            predictions = (probs >= threshold).astype(np.float32)

            # 完全匹配准确率 (所有标签都正确)
            exact_match = (predictions == labels).all(axis=1).mean()

            # 子集准确率 (至少有一个标签正确且没有假阳性?)
            # 使用更实用的: 按样本平均准确率
            sample_acc = (predictions == labels).mean()

            # 每个标签的 F1
            per_label = {}
            for i, label_name in enumerate(SUB_AGENT_LABELS):
                tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum()
                fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum()
                fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                per_label[f"{label_name}_f1"] = f1

            return {
                "exact_match": exact_match,
                "sample_accuracy": sample_acc,
                **per_label,
            }

        # 6. 训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print(f"[Trainer] 多标签训练开始... "
              f"(epochs={epochs}, batch_size={batch_size}, lr={learning_rate})")
        trainer.train()

        # 7. 评估
        eval_results = trainer.evaluate()
        print(f"[Trainer] 验证集评估: {eval_results}")

        # 8. 保存模型
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"[Trainer] 模型已保存至: {self.output_dir}")

        return {
            "eval_loss": eval_results.get("eval_loss"),
            "eval_exact_match": eval_results.get("eval_exact_match"),
            "eval_sample_accuracy": eval_results.get("eval_sample_accuracy"),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "output_dir": self.output_dir,
        }

    # ===================== 评估 =====================

    def evaluate(self, model_path: str = None, test_data_path: str = None) -> Dict:
        """
        在测试集上评估多标签模型。

        Args:
            model_path: 模型路径
            test_data_path: 测试数据路径

        Returns:
            评估指标
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_path = model_path or self.output_dir
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(router_config.device)
        model.eval()

        data_path = test_data_path or self.annotations_path
        data = self.load_annotations() if Path(data_path).exists() else []

        if not data:
            return {"error": "没有可评估的数据"}

        threshold = router_config.multi_label_threshold
        correct_exact = 0
        total = len(data)
        label_tp = {l: 0 for l in SUB_AGENT_LABELS}
        label_fp = {l: 0 for l in SUB_AGENT_LABELS}
        label_fn = {l: 0 for l in SUB_AGENT_LABELS}

        for item in data:
            query = item["query"]
            true_labels = set(item["labels"])

            inputs = tokenizer(
                query,
                max_length=router_config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(router_config.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()

            pred_labels = set()
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    pred_labels.add(self._id_to_label[i])

            # 完全匹配
            if pred_labels == true_labels:
                correct_exact += 1

            # 每个标签的混淆矩阵
            for label in SUB_AGENT_LABELS:
                in_pred = label in pred_labels
                in_true = label in true_labels
                if in_pred and in_true:
                    label_tp[label] += 1
                elif in_pred and not in_true:
                    label_fp[label] += 1
                elif not in_pred and in_true:
                    label_fn[label] += 1

        accuracy = correct_exact / total if total > 0 else 0

        per_label_metrics = {}
        for label in SUB_AGENT_LABELS:
            tp, fp, fn = label_tp[label], label_fp[label], label_fn[label]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_label_metrics[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

        print(f"[Trainer] 测试集完全匹配率: {accuracy:.4f}")
        print(f"[Trainer] 各类别指标: {per_label_metrics}")

        return {
            "exact_match_accuracy": accuracy,
            "per_label_metrics": per_label_metrics,
            "total_samples": total,
        }


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(description="BERT 路由模型训练 (多标签模式)")
    parser.add_argument("--data", type=str, default=None,
                        help="标注数据 JSON 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估不训练")
    args = parser.parse_args()

    trainer = BertRouterTrainer(
        annotations_path=args.data,
        output_dir=args.output,
    )

    if args.eval_only:
        trainer.evaluate()
    else:
        result = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        print(f"\n训练完成: {result}")


if __name__ == "__main__":
    main()
