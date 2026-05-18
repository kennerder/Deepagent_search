"""
父子文档切分器 (Parent-Child Document Splitter)

策略（参考掌柜智库项目）:
  1. Markdown 标题优先切分：按 # 标题层级识别逻辑章节边界
  2. 父块 (Parent): 较大的上下文块 (2048 tokens)，返回给 LLM 时使用
  3. 子块 (Child): 较小的检索块 (512 tokens)，存入 Qdrant 用于向量检索
  4. 关联机制: 每个子块记录其所属父块的 ID，检索命中子块时返回对应父块

切分流程:
  MD 标题初切 → 超长递归切 (段落→句子) → 短块合并 → 建立父子关联
"""

import re
import uuid
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import rag_config


class ParentChildSplitter:
    """
    父子文档切分器

    子块用于向量检索（精匹配），父块用于上下文返回（完整性）。
    检索流程: Query → 召回子块 → 追溯父块 → 返回父块内容给 LLM
    """

    def __init__(self):
        self.child_size = rag_config.child_chunk_size
        self.child_overlap = rag_config.child_chunk_overlap
        self.parent_size = rag_config.parent_chunk_size
        self.parent_overlap = rag_config.parent_chunk_overlap
        self.min_length = rag_config.min_chunk_length
        self.use_headers = rag_config.use_markdown_headers

    def split(self, content: str, file_name: str = "") -> Dict[str, Any]:
        """
        主入口：对文档执行父子切分。

        Args:
            content: 文档原始文本
            file_name: 文件名（用于溯源）

        Returns:
            {
                "parent_chunks": [{"id", "content", "title", "file_name"}, ...],
                "child_chunks":  [{"id", "content", "parent_id", "title", "file_name"}, ...],
            }
        """
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # 阶段1: 按 Markdown 标题粗切
        sections = self._split_by_titles(content, file_name)

        # 阶段2: 生成父块（大块）
        parent_chunks = self._create_parents(sections, file_name)

        # 阶段3: 生成子块（小块），建立父子关联
        child_chunks = self._create_children(parent_chunks, file_name)

        return {
            "parent_chunks": parent_chunks,
            "child_chunks": child_chunks,
        }

    # ========================= 阶段1：标题切分 =========================

    def _split_by_titles(self, content: str, file_name: str) -> List[Dict]:
        """按 Markdown 标题语法进行第一轮粗切分"""
        lines = content.split("\n")
        sections = []
        current_title = file_name
        current_lines = []
        in_code_block = False
        title_pattern = r'^\s*#{1,6}\s+.+'

        for line in lines:
            stripped = line.strip()

            # 跳过代码块内的 # 伪标题
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            is_title = (not in_code_block) and re.match(title_pattern, line)
            if is_title:
                if current_lines:
                    sections.append({
                        "title": current_title,
                        "content": "\n".join(current_lines),
                    })
                current_title = stripped
                current_lines = [line]
            else:
                current_lines.append(line)

        # 最后一个章节
        if current_lines:
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),
            })

        return sections

    # ========================= 阶段2：创建父块 =========================

    def _create_parents(self, sections: List[Dict], file_name: str) -> List[Dict]:
        """从章节生成父块（大块，用于 LLM 上下文）"""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        parents = []
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            if not content.strip():
                continue

            chunks = parent_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                parent_id = str(uuid.uuid4())
                parents.append({
                    "id": parent_id,
                    "content": f"{title}\n\n{chunk}" if title else chunk,
                    "title": f"{title}#{i}" if title else f"p-{i}",
                    "file_name": file_name,
                    "section_title": title,
                    "part": i,
                })

        return parents

    # ========================= 阶段3：创建子块 & 父子关联 =========================

    def _create_children(self, parents: List[Dict], file_name: str) -> List[Dict]:
        """从父块生成子块，每个子块关联父块 ID"""
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        children = []
        for parent in parents:
            parent_id = parent["id"]
            parent_content = parent["content"]

            sub_chunks = child_splitter.split_text(parent_content)
            for j, sub_chunk in enumerate(sub_chunks):
                if len(sub_chunk.strip()) < self.min_length and len(sub_chunks) > 1:
                    continue  # 跳过过短的子块（非单一块情况）

                child_id = str(uuid.uuid4())
                children.append({
                    "id": child_id,
                    "content": sub_chunk,
                    "parent_id": parent_id,
                    "title": f"{parent['title']}-c{j}",
                    "file_name": file_name,
                    "parent_title": parent.get("section_title", ""),
                    "part": j,
                })

        return children


def create_splitter() -> ParentChildSplitter:
    """工厂函数"""
    return ParentChildSplitter()
