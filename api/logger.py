import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class AgentLogger:
    """Agent 日志记录器，按 thread_id 分类归档日志"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._loggers: Dict[str, logging.Logger] = {}

    def _get_logger(self, thread_id: str) -> logging.Logger:
        if thread_id not in self._loggers:
            logger = logging.getLogger(f"agent.{thread_id}")
            logger.setLevel(logging.DEBUG)

            log_file = self.log_dir / f"{thread_id}.log"
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(handler)
            self._loggers[thread_id] = logger

        return self._loggers[thread_id]

    def log_main_chunk(self, chunk: Dict[str, Any], thread_id: Optional[str] = None):
        """记录主智能体的流式输出 chunk"""
        logger = self._get_logger(thread_id or "default")
        logger.debug(f"Chunk: {json.dumps(chunk, ensure_ascii=False, default=str)}")

    def log_tool_call(self, tool_name: str, args: Dict, thread_id: Optional[str] = None):
        """记录工具调用"""
        logger = self._get_logger(thread_id or "default")
        logger.info(f"Tool Call: {tool_name}, Args: {json.dumps(args, ensure_ascii=False, default=str)}")

    def log_error(self, error: str, thread_id: Optional[str] = None):
        """记录错误"""
        logger = self._get_logger(thread_id or "default")
        logger.error(f"Error: {error}")


class AgentLogCallbackHandler:
    """LangChain 回调处理器，用于集成到 Agent 执行链路中"""

    def __init__(self, logger: AgentLogger, thread_id: str = "default"):
        self.logger = logger
        self.thread_id = thread_id

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        self.logger.log_tool_call(
            serialized.get("name", "unknown"),
            {"input": input_str},
            self.thread_id
        )

    def on_tool_error(self, error: Exception, **kwargs):
        self.logger.log_error(str(error), self.thread_id)


# 全局单例
agent_logger = AgentLogger()
