import os
import re
import sys
import uuid
import asyncio
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import shutil

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import agent runner and monitor
from agent.main_agent import run_deep_agent
from api.monitor import monitor, manager

app = FastAPI(title="DeepAgents API")

# 会话 ID 会参与目录路径拼接，只允许安全字符
SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# 可选鉴权：设置 API_TOKEN 后，REST 需带 X-API-Token 头，WebSocket 需带 ?token=
API_TOKEN = os.getenv("API_TOKEN", "").strip()

if API_TOKEN:
    @app.middleware("http")
    async def require_api_token(request, call_next):
        if request.url.path.startswith("/api/") and request.headers.get("x-api-token") != API_TOKEN:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

# 挂载输出目录，以便前端访问生成的静态文件
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# 定义上传目录 updated
updated_dir = project_root / "updated"
updated_dir.mkdir(exist_ok=True)

# 配置 CORS：默认放开来源但不允许携带凭证；
# 需要跨域带 Cookie 时用 CORS_ORIGINS 显式列出来源（逗号分隔）
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    query: str
    thread_id: str = None


@app.post("/api/task")
async def run_task(request: TaskRequest):
    """
    智能体任务启动接口 (Run Agent Task)。

    目标：
    1. 接收用户的自然语言指令。
    2. 在后台异步启动 Agent 执行逻辑。
    3. 返回会话 ID，供前端通过 WebSocket 订阅实时进度。

    Args:
        request (TaskRequest): 包含用户 query 和可选 thread_id 的请求体。
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    # 后台异步运行 Agent，不阻塞主线程
    asyncio.create_task(run_deep_agent(request.query, thread_id))

    return {"status": "started", "thread_id": thread_id}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), thread_id: str = Form(...)):
    """
    文件上传接口 (File Upload)。

    目标：
    1. 接收用户上传的一个或多个文件。
    2. 保存到 `updated/session_{thread_id}` 目录。
    3. 供 Agent 在后续任务中读取和分析。

    Args:
        files (List[UploadFile]): 文件对象列表。
        thread_id (str): 关联的任务会话 ID。
    """
    # thread_id 与文件名都来自请求体，必须清洗后再拼路径，否则 ../ 可以逃出上传目录
    if not SESSION_ID_RE.fullmatch(thread_id or ""):
        return JSONResponse({"error": "无效的 thread_id"}, status_code=400)

    target_dir = (updated_dir / f"session_{thread_id}").resolve()
    if not target_dir.is_relative_to(updated_dir.resolve()):
        return JSONResponse({"error": "拒绝访问: 目录越界"}, status_code=400)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        safe_name = Path(file.filename or "").name  # 只保留文件名，丢掉路径成分
        if not safe_name or safe_name in {".", ".."}:
            continue
        file_path = (target_dir / safe_name).resolve()
        if not file_path.is_relative_to(target_dir):
            continue
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(safe_name)

    return {"status": "uploaded", "files": saved_files}


@app.get("/api/download")
async def download_file(path: str):
    """
    文件下载接口 (File Download)。

    目标：
    1. 根据绝对路径下载文件。
    2. 严格的安全检查，防止越权访问。

    Args:
        path (str): 文件的绝对路径 (通常从 list_files 接口获取)。
    """
    try:
        abs_path = Path(path).resolve()
        output_abs = output_dir.resolve()

        if not abs_path.is_relative_to(output_abs):
            return {"error": "拒绝访问: 只能下载输出目录下的文件"}
    except Exception:
        return {"error": "无效的路径参数"}

    if not abs_path.exists():
        return {"error": "文件不存在"}

    return FileResponse(abs_path, filename=abs_path.name)


@app.get("/api/files")
async def list_files(path: str):
    """
    文件列表查询接口 (File Explorer)。

    目标：
    1. 列出指定目录下的所有生成文件。
    2. 提供文件元数据（大小、时间、下载链接）。
    3. 严格的安全检查，防止路径遍历攻击。

    Args:
        path (str): 目标目录的绝对路径 (必须在 output 目录下)。
    """
    print(f"[DEBUG] 请求文件列表: {path}")

    try:
        abs_path = Path(path).resolve()
        output_abs = output_dir.resolve()

        if not abs_path.is_relative_to(output_abs):
            print(f"[ERROR] 拒绝访问: {abs_path} 不在 {output_abs} 目录下")
            return {"error": "拒绝访问: 只能访问输出目录下的文件"}

    except Exception as e:
        print(f"[ERROR] 路径解析失败: {e}")
        return {"error": f"路径无效: {e}"}

    if not abs_path.exists():
        return {"error": "目录不存在"}

    files = []
    try:
        for file_path in abs_path.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "type": "file",
                    "path": str(file_path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime
                })

    except Exception as e:
        print(f"[ERROR] 遍历文件失败: {e}")
        return {"error": str(e)}

    files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    print(f"[DEBUG] 找到 {len(files)} 个文件")
    return {"files": files}


@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """
    WebSocket 实时通讯核心接口 (Real-time Communication)。

    目标：
    1. 建立长连接，实现服务端与前端的双向通信。
    2. 绑定 `thread_id`，实现会话级消息隔离。
    3. 维持心跳 (Keep-Alive)，防止连接超时。

    Args:
        websocket (WebSocket): WebSocket 连接实例。
        thread_id (str): 当前会话的唯一标识。
    """
    if API_TOKEN and websocket.query_params.get("token") != API_TOKEN:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, thread_id)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "pong",
                "message": f"服务端已收到: {data}"
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, thread_id)
        print(f"[WebSocket] 客户端已断开: {thread_id}")

    except Exception as e:
        print(f"[WebSocket] 连接异常: {e}")
        manager.disconnect(websocket, thread_id)


# 挂载 Vue 构建产物 (生产模式)
ui_dist = project_root / "ui" / "dist"
if ui_dist.exists():
    assets_dir = ui_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """SPA 回退：非 API 路径返回 index.html"""
        index_path = ui_dist / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Frontend not built</h1>", status_code=404)


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
