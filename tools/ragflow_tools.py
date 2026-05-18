# 导入系统核心模块
import os
import logging
# 导入自定义监控模块（用于上报工具调用日志）
from api.monitor import monitor
# 导入HTTP请求库（用于健康检查）
import requests
# 导入RAGFlow SDK核心类（用于操作RAGFlow助手/知识库）
from ragflow_sdk import RAGFlow
# 导入环境变量加载工具（用于读取.env文件中的配置）
from dotenv import load_dotenv
# 导入LangChain工具装饰器（用于将函数注册为Agent可调用的工具）
from langchain_core.tools import tool
from typing_extensions import Annotated

# 初始化日志器
logger = logging.getLogger(__name__)

from typing import Tuple, Optional


def _load_ragflow_env() -> Tuple[Optional[str], Optional[str]]:
    """
    加载RAGFlow的环境变量（API密钥和服务地址）
    优先加载当前脚本目录下的.env文件，若不存在则加载系统环境变量

    Returns:
        Tuple[Optional[str], Optional[str]]:
            - 第一个值：RAGFlow API密钥（RAGFLOW_API_KEY）
            - 第二个值：RAGFlow服务地址（RAGFLOW_API_URL）
            - 若未配置则返回None
    """
    load_dotenv()

    api_key = os.getenv("RAGFLOW_API_KEY")
    base_url = os.getenv("RAGFLOW_API_URL")
    return api_key, base_url


@tool
def get_assistant_list(
    dummy_arg: Annotated[str, "不需要输入参数，直接调用即可"] = "",
) -> str:
    """
    【工具功能】获取 RAGFlow 中所有聊天助手信息
    适用场景：Agent 需要确认当前有哪些可用助手，及每个助手绑定的知识库范围时调用
    返回：结构化字符串（助手名称+功能介绍+关联知识库）
    """
    monitor.report_tool("RAGFlow助手列表查询")
    api_key, base_url = _load_ragflow_env()

    if not api_key or not base_url:
        return "错误：RAGFlow 环境变量未配置（需设置 RAGFLOW_API_URL 与 RAGFLOW_API_KEY）"

    result = ""
    try:
        rag = RAGFlow(api_key=api_key, base_url=base_url)
        for assistant in rag.list_chats():
            kb_names = []
            if assistant.datasets and isinstance(assistant.datasets, list):
                for dataset in assistant.datasets:
                    if isinstance(dataset, dict) and "name" in dataset:
                        kb_names.append(dataset["name"])

            kb_names_str = "、".join(kb_names) if kb_names else "无"
            result += f"助手名称：{assistant.name}； 功能介绍：{assistant.description}； 关联知识库：{kb_names_str}\n"

        return result.rstrip("\n") if result else "未找到任何聊天助手"
    except Exception as e:
        return f"获取助手列表失败：{str(e)}"


@tool
def create_ask_delete(
    assistant_name: Annotated[str, "必填：目标聊天助手的名称"],
    question: Annotated[str, "必填：要向助手提问的问题"],
) -> str:
    """
    【工具功能】向指定 RAGFlow 助手发起单次提问（临时会话，用完即删）
    适用场景：Agent 需单次查询某个助手，无需保留会话记录时调用
    特点：创建临时会话→流式接收答案→自动删除会话，无数据残留
    """
    monitor.report_tool(
        "RAGFlow助手提问工具",
        {"助手名称": assistant_name, "查询问题": question}
    )
    api_key, base_url = _load_ragflow_env()

    try:
        rag = RAGFlow(api_key=api_key, base_url=base_url)

        assistants = rag.list_chats(name=assistant_name)
        if not assistants:
            return f"错误：未找到名为「{assistant_name}」的聊天助手"
        assistant = assistants[0]

        session = None
        try:
            session = assistant.create_session(name="temp_session_for_single_ask")

            response_generator = session.ask(question, stream=True)

            full_answer = ""
            for part in response_generator:
                if hasattr(part, "content") and part.content:
                    full_answer = part.content

            monitor.report_tool(
                "RAGFlow助手回答记录",
                {"助手名称": assistant_name, "问题": question, "答案": full_answer}
            )

            if session and hasattr(session, "id"):
                assistant.delete_sessions(ids=[session.id])

            return full_answer if full_answer else "未获取到助手的回答"

        except Exception as e:
            return f"提问过程失败：{str(e)}"

    except Exception as e:
        return f"RAGFlow 操作失败：{str(e)}"
