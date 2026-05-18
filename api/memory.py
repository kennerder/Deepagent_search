"""
对话记忆管理器 (Conversation Memory)

两层记忆:
  - 短期: 按 thread_id 存储最近 N 轮对话，用于多轮上下文关联和指代消解
  - 长期: 预留接口 (用户偏好、常用查询模式等)

线程安全: asyncio.Lock 保护每个 thread_id 的写操作

使用方式:
  from api.memory import memory_manager

  # 存储一轮对话
  memory_manager.add_turn(thread_id, user_query, agent_response)

  # 获取上下文 (用于下一轮 prompt 构建)
  context = memory_manager.get_context(thread_id)

  # 获取原始历史 (用于指代消解)
  history = memory_manager.get_history(thread_id)
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_query: str
    agent_response: str


@dataclass
class ThreadMemory:
    """单个会话的记忆"""
    turns: List[ConversationTurn] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConversationMemory:
    """
    对话记忆管理器（进程内单例）

    按 thread_id 隔离，每个会话独立记忆空间。
    """

    def __init__(self, max_turns: int = 20):
        self._store: Dict[str, ThreadMemory] = {}
        self._max_turns = max_turns  # 每个会话最多保留的轮数
        self._global_lock = asyncio.Lock()

    async def _get_or_create(self, thread_id: str) -> ThreadMemory:
        """获取或创建会话记忆"""
        if thread_id not in self._store:
            async with self._global_lock:
                if thread_id not in self._store:
                    self._store[thread_id] = ThreadMemory()
        return self._store[thread_id]

    async def add_turn(
        self,
        thread_id: str,
        user_query: str,
        agent_response: str,
    ):
        """
        存储一轮对话。

        Args:
            thread_id: 会话 ID
            user_query: 用户原始查询
            agent_response: Agent 最终回复
        """
        memory = await self._get_or_create(thread_id)
        async with memory.lock:
            memory.turns.append(ConversationTurn(
                user_query=user_query,
                agent_response=agent_response,
            ))
            # 超过最大轮数时裁剪旧的
            if len(memory.turns) > self._max_turns:
                memory.turns = memory.turns[-self._max_turns:]

    async def get_context(
        self,
        thread_id: str,
        max_turns: int = 5,
    ) -> str:
        """
        获取会话上下文，用于注入 prompt。

        返回最近 N 轮对话的摘要，格式化为自然语言。

        Args:
            thread_id: 会话 ID
            max_turns: 最多返回最近 N 轮

        Returns:
            格式化的上下文字符串，无历史时返回空字符串
        """
        if thread_id not in self._store:
            return ""

        memory = self._store[thread_id]
        async with memory.lock:
            recent = memory.turns[-max_turns:]

        if not recent:
            return ""

        lines = ["【对话历史】以下是本轮之前用户与助手的对话记录，可用于指代消解和上下文理解:"]
        for i, turn in enumerate(recent, 1):
            lines.append(f"  第{i}轮 - 用户: {turn.user_query}")
            # 截断过长的回复
            response_summary = turn.agent_response[:200]
            if len(turn.agent_response) > 200:
                response_summary += "..."
            lines.append(f"  第{i}轮 - 助手: {response_summary}")

        return "\n".join(lines)

    async def get_history(
        self,
        thread_id: str,
        max_turns: int = 5,
    ) -> List[Dict]:
        """
        获取原始对话历史，用于指代消解等需要结构化数据的场景。

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        if thread_id not in self._store:
            return []

        memory = self._store[thread_id]
        async with memory.lock:
            recent = memory.turns[-max_turns:]

        history = []
        for turn in recent:
            history.append({"role": "user", "content": turn.user_query})
            history.append({"role": "assistant", "content": turn.agent_response})
        return history

    async def clear(self, thread_id: str):
        """清除某个会话的记忆"""
        async with self._global_lock:
            self._store.pop(thread_id, None)

    async def get_turn_count(self, thread_id: str) -> int:
        """获取会话轮数"""
        if thread_id not in self._store:
            return 0
        memory = self._store[thread_id]
        async with memory.lock:
            return len(memory.turns)

    def get_active_sessions(self) -> List[str]:
        """获取所有活跃会话 ID"""
        return list(self._store.keys())


# 全局单例
memory_manager = ConversationMemory()
