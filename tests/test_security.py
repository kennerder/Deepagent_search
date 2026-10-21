# -*- coding: utf-8 -*-
"""安全回归测试：上传路径清洗、可选鉴权、数据库只读守卫。

运行：pytest tests/ -q
这些测试不需要 MySQL / Qdrant / LLM，重依赖在导入前被打桩。
"""
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 需要在导入 api.server 之前设置，鉴权中间件是导入期决定是否注册的
os.environ.setdefault("API_TOKEN", "test-token")
TOKEN_HEADER = {"X-API-Token": os.environ["API_TOKEN"]}

# 打桩 agent.main_agent，避免测试拉起 langchain / deepagents 全家桶
if "agent.main_agent" not in sys.modules:
    _agent_stub = types.ModuleType("agent.main_agent")

    async def run_deep_agent(query, thread_id):  # noqa: ARG001
        return None

    _agent_stub.run_deep_agent = run_deep_agent
    sys.modules["agent.main_agent"] = _agent_stub

from fastapi.testclient import TestClient  # noqa: E402

import api.server as server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


def test_api_requires_token(client):
    assert client.post("/api/task", json={"query": "hi"}).status_code == 401


def test_api_accepts_valid_token(client):
    resp = client.post("/api/task", json={"query": "hi"}, headers=TOKEN_HEADER)
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_upload_strips_path_components(client, tmp_path):
    """文件名里的 ../ 必须被剥掉，不能写到上传目录之外。"""
    resp = client.post(
        "/api/upload",
        files=[("files", ("../../../evil.py", b"pwn", "text/plain"))],
        data={"thread_id": "11111111-2222-3333-4444-555555555555"},
        headers=TOKEN_HEADER,
    )
    assert resp.status_code == 200
    assert resp.json()["files"] == ["evil.py"]
    assert not (REPO / "evil.py").exists()
    assert not (REPO.parent / "evil.py").exists()
    for stray in (REPO / "updated").glob("session_11111111*"):
        stray.unlink() if stray.is_file() else __import__("shutil").rmtree(stray, ignore_errors=True)


def test_upload_rejects_traversal_thread_id(client):
    resp = client.post(
        "/api/upload",
        files=[("files", ("a.txt", b"x", "text/plain"))],
        data={"thread_id": "../../etc"},
        headers=TOKEN_HEADER,
    )
    assert resp.status_code == 400


def test_websocket_requires_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/abc-123") as ws:
            ws.receive_text()


@pytest.mark.parametrize(
    "sql",
    ["SELECT 1", "select * from drugs", "WITH x AS (SELECT 1) SELECT * FROM x",
     "SHOW TABLES", "EXPLAIN SELECT 1", "SELECT 1;"],
)
def test_sql_guard_allows_read_only(sql):
    assert _mysql_tools().is_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    ["DROP TABLE drugs", "DELETE FROM drugs", "UPDATE drugs SET a = 1",
     "INSERT INTO t VALUES (1)", "SELECT 1; DROP TABLE drugs"],
)
def test_sql_guard_blocks_writes(sql):
    assert not _mysql_tools().is_read_only(sql)


def _mysql_tools():
    """导入 tools.mysql_tools，按需为其缺失的重依赖打桩。"""
    if "mysql.connector" not in sys.modules:
        sys.modules["mysql.connector"] = types.SimpleNamespace(connect=None, Error=Exception)
    if "langchain_core" not in sys.modules:
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    if "langchain_core.tools" not in sys.modules:
        sys.modules["langchain_core.tools"] = types.SimpleNamespace(tool=lambda f: f)
    import tools.mysql_tools as module

    return module
