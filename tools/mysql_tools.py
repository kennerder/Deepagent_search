import os
from dotenv import load_dotenv
from api.monitor import monitor
from mysql.connector import connect, Error
from typing import Annotated, List
from langchain_core.tools import tool

load_dotenv()

# 只读语句白名单：LLM 生成的 SQL 默认只能查询，禁止 DDL/DML
READ_ONLY_PREFIXES = ("select", "with", "show", "desc", "describe", "explain")


def is_read_only(query: str) -> bool:
    """仅放行只读语句；本地调试需要写操作时设置 ALLOW_WRITE_SQL=true。"""
    if os.getenv("ALLOW_WRITE_SQL", "false").strip().lower() == "true":
        return True
    stripped = query.strip().lstrip("(").strip()
    if not stripped or not stripped.lower().startswith(READ_ONLY_PREFIXES):
        return False
    return all(not part.strip() for part in query.strip().split(";")[1:])


# 加载配置文件方便后续使用
def get_db_config():
    """Get database configuration from environment variables."""
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL")
    }
    # 移除 None 值
    config = {k: v for k, v in config.items() if v is not None}

    # 校验核心配置是否存在
    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config


@tool
def list_sql_tables() -> Annotated[str, "数据库中可用的表名列表，以逗号分隔"]:
    """
    列出配置的 MySQL 数据库中所有可用的表。
    核心用途：
        AI Agent 需要查看数据库中有哪些表时调用，为后续执行 SQL 查询提供基础信息。
    返回值：
        str: 成功时返回 "可用数据表：表1, 表2, ..."；
             配置缺失时返回错误提示；
             执行异常时返回具体错误信息。
    异常处理：
        捕获数据库连接/执行 SQL 时的所有 Error 异常，返回可读的错误信息，避免 Agent 崩溃。
    """
    monitor.report_tool("数据库表获取工具")
    config = get_db_config()
    try:
        if not all([config.get("user"), config.get("password"), config.get("database")]):
            return "错误：数据库配置缺失（缺少MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE配置项）。"
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                if not tables:
                    return "数据库中未找到任何数据表。"
                table_names = [table[0] for table in tables]
                return f"可用数据表：{', '.join(table_names)}"
    except Error as e:
        return f"列出数据表失败：{str(e)}"


@tool
def get_table_data(
        table_name: Annotated[str, "要读取数据的表名"]
) -> Annotated[str, "表的前 100 行数据（CSV 格式）"]:
    """
    读取指定 MySQL 数据表的前 100 行数据，返回 CSV 格式结果。
    """
    monitor.report_tool("数据库内容浏览工具", {"正在读取的表": table_name})
    config = get_db_config()

    try:
        if not all([config.get("user"), config.get("password"), config.get("database")]):
            return "错误：数据库配置缺失（请检查账号、密码、数据库名）。"

        with connect(**config) as conn:
            with conn.cursor() as cursor:
                # 表名拼进 SQL 前必须比对 SHOW TABLES 白名单，字符串过滤挡不住注入
                cursor.execute("SHOW TABLES")
                known_tables = {row[0] for row in cursor.fetchall()}
                if table_name not in known_tables:
                    return f"错误：数据表 {table_name} 不存在"

                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 100")

                if cursor.description is None:
                    return f"数据表 {table_name} 为空或表名无效。"

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                result = [",".join(map(str, row)) for row in rows]
                header = ",".join(columns)

                return f"{header}\n" + "\n".join(result)

    except Error as e:
        return f"读取数据表 {table_name} 失败：{str(e)}"


@tool
def execute_sql_query(
        query: Annotated[str, "要执行的 SQL 查询语句"]
) -> Annotated[str, "查询结果或成功消息"]:
    """在 MySQL 数据库上执行自定义 SQL 查询。用于复杂查询、联接或特定数据检索。"""
    monitor.report_tool("数据库查询工具")
    if not is_read_only(query):
        return "错误：仅允许执行只读查询（SELECT / WITH / SHOW / DESC / EXPLAIN）"
    config = get_db_config()
    try:
        if not all([config.get("user"), config.get("password"), config.get("database")]):
            return "错误：数据库配置缺失（请检查账号、密码、数据库名）。"
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)

                if cursor.description is not None:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    if not rows:
                        return f"查询执行成功，无数据返回。涉及列名：{', '.join(columns)}"

                    result_lines = []
                    result_lines.append(",".join(columns))
                    for row in rows:
                        result_lines.append(",".join(map(str, row)))

                    return "\n".join(result_lines)

                else:
                    return f"SQL 执行成功，受影响行数：{cursor.rowcount}"

    except Error as e:
        return f"执行 SQL 失败：{str(e)}"
