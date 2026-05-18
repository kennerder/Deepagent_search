"""
BERT 意图分类器

模型架构:
  hfl/chinese-roberta-wwm-ext (预训练 backbone)
  → Dropout(0.1)
  → Linear(768 → 3)  分类头

推理模式:
  - 单标签 (argmax): 取最高分标签, 分数 >= threshold 才生效
  - 多标签 (sigmoid): 每个标签独立判断 threshold

使用方式:
  classifier = BertIntentClassifier()
  classifier.load()                           # 加载模型
  result = classifier.predict("查询布洛芬库存")  # → {"label": "db_query", "score": 0.92}
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from router.config import router_config, SUB_AGENT_LABELS


class BertIntentClassifier:
    """
    BERT 意图分类器（单例模式）

    hfl/chinese-roberta-wwm-ext:
      - 12-layer, 768-hidden, 12-heads, 110M parameters
      - 全词掩码预训练，中文 NLU 表现优于原始 BERT
    """

    _instance: Optional["BertIntentClassifier"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance._loaded = False
        return cls._instance

    # ===================== 加载 =====================

    def load(self, model_path: Optional[str] = None):
        """
        加载模型和分词器。

        优先级:
          1. 指定 model_path → 加载微调模型
          2. finetuned_model_dir 存在 → 加载微调模型
          3. 否则 → 加载预训练模型 (未微调, 需先训练)

        Args:
            model_path: 模型目录路径 (可选)
        """
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # 确定加载路径
        if model_path:
            load_path = model_path
        elif Path(router_config.finetuned_model_dir).exists():
            load_path = router_config.finetuned_model_dir
        else:
            load_path = router_config.model_name
            print(f"[Classifier] 加载预训练模型 (未微调): {load_path}")
            print(f"[Classifier] 警告: 未微调模型输出随机，请先执行 trainer.py 进行微调")

        # 加载分词器和模型
        self._tokenizer = AutoTokenizer.from_pretrained(load_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=router_config.num_labels,
            problem_type="multi_label_classification",
        )
        self._model.to(router_config.device)
        self._model.eval()
        self._loaded = True

        print(f"[Classifier] 模型加载完成: {load_path}")

    def is_loaded(self) -> bool:
        return self._loaded

    # ===================== 推理 =====================

    def predict(
        self,
        query: str,
        return_all_scores: bool = False,
    ) -> Union[Dict, List[Dict]]:
        """
        单条查询推理。

        Args:
            query: 用户查询文本
            return_all_scores: 是否返回所有标签的分数

        Returns:
            单标签模式: {"label": str, "score": float, "hit": bool}
            多标签模式: {"labels": [str], "scores": [float]}
            return_all_scores: [{"label": str, "score": float}, ...]
        """
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 classifier.load()")

        import torch
        import torch.nn.functional as F

        # 分词
        inputs = self._tokenizer(
            query,
            max_length=router_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(router_config.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]  # (num_labels,)

        if router_config.multi_label:
            # 多标签模式: sigmoid + threshold
            probs = torch.sigmoid(logits).cpu().numpy()
            result = self._decode_multi_label(probs)

        else:
            # 单标签模式: softmax + argmax
            probs = F.softmax(logits, dim=0).cpu().numpy()
            result = self._decode_single_label(probs)

        if return_all_scores:
            all_scores = [
                {"label": SUB_AGENT_LABELS[i], "score": float(probs[i])}
                for i in range(len(SUB_AGENT_LABELS))
            ]
            all_scores.sort(key=lambda x: x["score"], reverse=True)
            return all_scores

        return result

    def predict_batch(
        self,
        queries: List[str],
    ) -> List[Dict]:
        """
        批量推理。

        Args:
            queries: 查询文本列表

        Returns:
            预测结果列表
        """
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 classifier.load()")

        import torch
        import torch.nn.functional as F

        # 批量分词
        inputs = self._tokenizer(
            queries,
            max_length=router_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(router_config.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (batch, num_labels)

        results = []
        for i in range(len(queries)):
            if router_config.multi_label:
                probs = torch.sigmoid(logits[i]).cpu().numpy()
                results.append(self._decode_multi_label(probs))
            else:
                probs = F.softmax(logits[i], dim=0).cpu().numpy()
                results.append(self._decode_single_label(probs))

        return results

    # ===================== 解码 =====================

    def _decode_single_label(self, probs: np.ndarray) -> Dict:
        """单标签解码: argmax + threshold"""
        top_idx = int(np.argmax(probs))
        top_score = float(probs[top_idx])
        threshold = router_config.single_label_threshold

        return {
            "label": SUB_AGENT_LABELS[top_idx] if top_score >= threshold else None,
            "score": top_score,
            "hit": top_score >= threshold,
            "candidate_label": SUB_AGENT_LABELS[top_idx],  # 始终保留最大概率标签
        }

    def _decode_multi_label(self, probs: np.ndarray) -> Dict:
        """多标签解码: sigmoid + threshold，不强制 fallback"""
        threshold = router_config.multi_label_threshold
        active = probs >= threshold

        labels = [SUB_AGENT_LABELS[i] for i, a in enumerate(active) if a]
        scores = [float(probs[i]) for i, a in enumerate(active) if a]

        # 无标签命中时 hit=False，触发 LLM fallback（不做 argmax 强制路由）
        return {
            "labels": labels,
            "scores": scores,
            "hit": len(labels) > 0,
        }

    # ===================== 保存 =====================

    def save(self, output_dir: Optional[str] = None):
        """保存微调后的模型"""
        if not self._loaded:
            raise RuntimeError("没有已加载的模型")

        output_dir = output_dir or router_config.finetuned_model_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        print(f"[Classifier] 模型已保存至: {output_dir}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self._loaded:
            return {"status": "not_loaded"}

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        return {
            "status": "loaded",
            "model_name": router_config.model_name,
            "num_labels": router_config.num_labels,
            "labels": SUB_AGENT_LABELS,
            "device": router_config.device,
            "multi_label": router_config.multi_label,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


# 全局单例
_classifier: Optional[BertIntentClassifier] = None


def get_classifier() -> BertIntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BertIntentClassifier()
    return _classifier
