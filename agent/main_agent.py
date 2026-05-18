from agent.sub_agents.knowledge_base_agent import knowledge_base_agent
from agent.sub_agents.database_query_agent import database_query_agent
from agent.sub_agents.network_search_agent import network_search_agent

# main_agent tool导入
from tools.markdown_tools import generate_markdown
from tools.pdf_tools import convert_md_to_pdf
from tools.upload_file_read_tool import read_file_content

from deepagents import create_deep_agent

from agent.llm import model
from agent.prompts import main_agent_config

from api.monitor import monitor
import asyncio
import uuid
import shutil
from pathlib import Path

from api.context import set_session_context, reset_session_context, set_thread_context

from langchain_core.messages import AIMessage

from api.logger import AgentLogger, AgentLogCallbackHandler
from api.memory import memory_manager
from router.integration import get_router

# 项目根目录 (agent -> 项目根)
project_root = Path(__file__).parent.parent

# 1. 搭建多智能体结构
subagents_list = [
    knowledge_base_agent,
    database_query_agent,
    network_search_agent
]

# 创建主智能体
main_agent = create_deep_agent(
    model=model,
    subagents=subagents_list,
    tools=[generate_markdown, convert_md_to_pdf, read_file_content],
    system_prompt=main_agent_config["system_prompt"]
)


def _prepare_session_environment(thread_id: str):
    """
    初始化会话运行环境（会话文件夹,以及相对路径，上传文件的信息！）。
    目标：
    1. 创建独立的物理工作空间。
    2. 处理用户上传的文件。
    3. 生成供 Agent 和前端使用的路径上下文（提示词）。

    执行步骤：
    1. 创建绝对路径：`project_root/output/session_{uuid}`。
    2. 标准化路径：转换为 POSIX 风格 (`/`) 以兼容 LLM 和跨平台。
    3. 文件迁移：将 `updated/session_{uuid}` 中的文件复制到工作目录。
    4. 构造提示词：生成包含已上传文件列表的 Context 文本。

    Returns:
        tuple: (
            session_dir_str (str): 物理工作目录的绝对路径 (当前会话对应文件存储位置)。
            relative_session_dir (str): 相对于项目根目录的路径 (用于提示词)。
            uploaded_info (str): 注入到 Prompt 中的文件列表描述。
        )
    """
    # 1. [创建] 定义并创建会话的绝对输出路径
    session_dir = project_root / "output" / f"session_{thread_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 2. [标准化] 路径转为 POSIX 风格
    session_dir_str = str(session_dir).replace("\\", "/")

    # 3. [相对化] 获取相对路径
    relative_session_dir = str(session_dir.relative_to(project_root)).replace("\\", "/")

    # 4. [迁移] 检查并处理上传文件
    upload_dir = project_root / "updated" / f"session_{thread_id}"
    uploaded_info = ""

    if upload_dir.exists():
        files = [f.name for f in upload_dir.iterdir() if f.is_file()]

        if files:
            for f in files:
                shutil.copy2(upload_dir / f, session_dir / f)

            uploaded_info = (f"\n    [已上传文件] 已加载到工作目录:\n" +
                             "\n".join([f"    - {f}" for f in files]) +
                             "\n    请优先使用工具读取并参考这些文件。")

    return session_dir_str, relative_session_dir, uploaded_info


def _process_stream_chunk(chunk) -> str:
    """
    处理 LangGraph 流式输出的增量状态 (Stream Processing)。
    目标：
    1. 解析 Agent 的每一步思考和行动。
    2. 识别关键事件（工具调用、子 Agent 委派、最终回复）。
    3. 通过 Monitor 实时上报状态给前端。
    核心逻辑：
    - 监听 `tool_calls` -> 记录日志，若是 'task' 则上报子 Agent 状态。
    - 监听 `content` -> 若无工具调用，则视为 Agent 的最终回复。

    Args:
        chunk (dict): 增量状态字典，如 {"node_name": {"messages": [AIMessage(...)]}}

    Returns:
        最终回复文本片段（可能为空字符串）
    """
    final_text = ""
    for node_name, state in chunk.items():
        if not state or "messages" not in state:
            continue
        messages = state["messages"]
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                # Case 1: Agent 决定调用工具 (Tool Call)
                if last_msg.tool_calls:
                    for tool in last_msg.tool_calls:
                        # 特殊处理：如果是 'task' 工具，说明正在委派给子 Agent
                        if tool['name'] == 'task':
                            monitor.report_assistant(
                                tool['args'].get('subagent_type', 'Agent'),
                                {"desc": tool['args'].get('description')}
                            )
                # Case 2: Agent 生成最终回复 (Final Answer)
                elif last_msg.content:
                    final_text = str(last_msg.content)
                    monitor.report_task_result(final_text)

    return final_text


# ====================== 核心执行逻辑 ======================
async def run_deep_agent(task_query: str, thread_id: str = None):
    """
    DeepAgents 核心执行入口 (Agent Execution Runtime)。

    目标：
    1. 接收用户的自然语言任务。
    2. 准备独立的运行环境 (Workspace)。
    3. 启动 LangGraph 智能体，并通过流式 (Stream) 实时处理每一步。
    4. 确保上下文隔离和异常安全。

    执行步骤：
    1. ID 初始化：确保每个任务有唯一的 `thread_id`。
    2. 环境准备：创建目录、迁移文件、生成路径信息。
    3. 上下文绑定：将 `thread_id` 和 `session_dir` 绑定到当前线程 (ContextVar)。
    4. 提示词构建：将环境信息注入到 Prompt。
    5. 流式执行：驱动 LangGraph 运行，并实时解析/上报每一个 Chunk。
    6. 资源清理：任务结束后（无论成功失败）重置上下文。
    """
    # 1. [ID 初始化] 确保有唯一的会话 ID
    if not thread_id:
        thread_id = str(uuid.uuid4())
    print(f"--- Start Task: {task_query} (Thread: {thread_id}) ---")

    # 2. [环境准备] 创建目录、处理上传文件
    session_dir_str, relative_session_dir, uploaded_info = _prepare_session_environment(thread_id)

    # 3. [上下文绑定] 初始化 ContextVars
    thread_token = set_thread_context(thread_id)
    session_token = set_session_context(session_dir_str)
    monitor.report_session_dir(session_dir_str)

    # 4. [运行时配置] LangChain Config
    config = {
        "configurable": {"thread_id": thread_id},
    }

    # 5. [BERT 路由] 意图分类 + 路由建议
    route_hint = ""
    try:
        router = get_router()
        route_result = await router.route(task_query, thread_id=thread_id)
        if route_result.route_hint:
            route_hint = "\n" + route_result.route_hint
            print(f"[Router] BERT 路由命中: {route_result.labels} "
                  f"(scores: {[f'{s:.2f}' for s in route_result.scores]})")
            monitor._emit("router_decision", f"路由决策: {route_result.labels}")
        else:
            print(f"[Router] 路由模式: {route_result.mode} (BERT 低置信度/未加载, LLM fallback)")
    except Exception as e:
        print(f"[Router] 路由异常, 降级为 LLM 路由: {e}")

    # 6. [对话记忆] 加载历史上下文
    memory_context = ""
    try:
        turn_count = await memory_manager.get_turn_count(thread_id)
        if turn_count > 0:
            memory_context = "\n" + await memory_manager.get_context(thread_id)
            print(f"[Memory] 加载会话历史: {turn_count} 轮")
    except Exception as e:
        print(f"[Memory] 加载异常: {e}")

    # 7. [提示词构建] 动态注入环境约束 + 历史上下文 + 路由建议
    path_instruction = f"""
    【工作环境指令】
    工作目录: {relative_session_dir}
    {uploaded_info}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir}/filename'
    2. 使用相对路径，禁止使用绝对路径
    3. 若存在上传文件，请先分析内容
    {memory_context}
    {route_hint}
    """

    # 8. [流式执行] 启动 Agent 循环
    final_response = ""
    try:
        async for chunk in main_agent.astream(
                {"messages": [{"role": "user", "content": task_query + path_instruction}]},
                config=config
        ):
            chunk_text = _process_stream_chunk(chunk)
            if chunk_text:
                final_response = chunk_text  # 持续覆盖，最终保留最后一条

        # 9. [记忆存储] 保存本轮对话
        if final_response:
            try:
                await memory_manager.add_turn(thread_id, task_query, final_response)
                print(f"[Memory] 已保存对话轮次: {thread_id}")
            except Exception as e:
                print(f"[Memory] 存储异常: {e}")

        return "Done"
    except Exception as e:
        print(f"Error: {e}")
        monitor._emit("error", f"Execution failed: {e}")

        # 异常时也保存部分记忆（避免丢失用户查询）
        try:
            await memory_manager.add_turn(thread_id, task_query, f"[执行异常] {str(e)}")
        except Exception:
            pass

        return f"Error: {e}"
    finally:
        # 10. [资源清理] 重置 ContextVars
        if 'session_token' in locals():
            reset_session_context(session_token, thread_token)


# ====================== 本地测试入口 ======================
if __name__ == "__main__":
    task = "查询数据库中的药品信息，生成一个pdf文件！"
    asyncio.run(run_deep_agent(task))
