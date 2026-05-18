"""
知识库构建 CLI 入口

用法:
  # 增量构建（默认）
  python -m rag_agent.scripts.build_kb

  # 强制重建
  python -m rag_agent.scripts.build_kb --rebuild

  # 指定文档目录
  python -m rag_agent.scripts.build_kb --dir /path/to/docs
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rag_agent.builder import KnowledgeBaseBuilder


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库构建工具")
    parser.add_argument("--dir", "-d", type=str, default=None,
                        help="文档目录路径 (默认: data/documents)")
    parser.add_argument("--rebuild", "-r", action="store_true",
                        help="强制重建（清空已有数据）")
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder()

    if args.rebuild:
        print(">>> 强制重建模式 <<<")
        result = builder.rebuild(docs_dir=args.dir)
    else:
        print(">>> 增量更新模式 <<<")
        result = builder.update(docs_dir=args.dir)

    print("\n构建结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
