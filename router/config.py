"""
BERT 路由模块配置

模型: hfl/chinese-roberta-wwm-ext (哈工大讯飞联合实验室)
  - 全词掩码 (WWM) 预训练，中文 NLU 效果优于原生 BERT
  - 110M 参数，微调成本低
  - HuggingFace 模型标识: hfl/chinese-roberta-wwm-ext
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


# 子代理标签定义
# BERT 只负责子代理路由，不涉及工具调用（工具调用仍由 LLM 负责）
SUB_AGENT_LABELS = [
    "web_search",    # 网络搜索助手 (Tavily)
    "db_query",      # 数据库查询助手 (SQL Agent)
    "kb_search",     # 知识库助手 (RAG)
]

# 标签中文映射
LABEL_CN_MAP = {
    "web_search": "网络搜索助手",
    "db_query": "数据库查询助手",
    "kb_search": "知识库助手",
}

# 标签对应的子代理名 (与 prompts.yml 中 sub_agents 的 name 对齐)
LABEL_SUB_AGENT_MAP = {
    "web_search": "网络搜索助手",
    "db_query": "数据库查询助手",
    "kb_search": "知识库助手",
}


@dataclass
class RouterConfig:
    """BERT 路由模块配置"""

    # --- 模型配置 ---
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    # 本地模型缓存路径 (首次使用自动下载到此目录)
    model_cache_dir: str = "models/bert_router"
    # 分类头输出维度
    num_labels: int = 3
    # 最大输入长度 (RoBERTa 限制 512)
    max_length: int = 128

    # --- 推理配置 ---
    # 推理设备: "cpu" | "cuda" | "cuda:0"
    device: str = "cpu"
    # 单标签模式阈值 (argmax 的 score >= 此值才路由，否则走 LLM fallback)
    single_label_threshold: float = 0.5
    # 多标签模式阈值 (sigmoid 输出 >= 此值视为命中)
    multi_label_threshold: float = 0.5
    # "无匹配"阈值: 全部标签得分均 < 此值时，判定为不需要任何子代理
    # 用于拦截"你好""今天天气"等与业务无关的闲聊
    none_threshold: float = 0.35
    # 是否启用多标签模式 (sigmoid 独立判断，支持同时激活多个子代理)
    # True: 多标签 sigmoid，允许同时路由到多个子代理（如"查库存+搜行情"→db_query+web_search）
    # False: 单标签 argmax，只取最高分（旧模式，不推荐）
    multi_label: bool = True

    # --- 训练配置 (见 trainer.py) ---
    # 微调后的模型保存路径
    finetuned_model_dir: str = "models/bert_router_finetuned"

    # --- 路由行为配置 ---
    # 置信度不足时的 fallback 策略: "llm" | "none"
    low_confidence_fallback: str = "llm"
    # 是否启用 LLM 改写 (指代消解) 后再送 BERT
    enable_query_rewrite: bool = True


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 全局实例
router_config = RouterConfig(
    model_cache_dir=str(PROJECT_ROOT / "models" / "bert_router"),
    finetuned_model_dir=str(PROJECT_ROOT / "models" / "bert_router_finetuned"),
)
