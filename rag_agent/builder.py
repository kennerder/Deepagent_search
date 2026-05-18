"""
知识库构建器 (Knowledge Base Builder)

构建流程:
  1. 加载文档: 从固定目录扫描新文档（MD5去重）
  2. 文档切分: 父子文档切分（子块512 tokens → 检索, 父块2048 tokens → 上下文）
  3. 向量化: BGE-M3 编码子块和父块
  4. 向量入库: 写入 Qdrant (子块集合 + 父块集合)
  5. 去重标记: 记录已处理文件

使用方式:
  # CLI
  python -m rag_agent.scripts.build_kb

  # 代码调用
  from rag_agent.builder import KnowledgeBaseBuilder
  builder = KnowledgeBaseBuilder()
  result = builder.build()
"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from rag_agent.config import rag_config, PROJECT_ROOT
from rag_agent.loader import DocumentLoader
from rag_agent.splitter import ParentChildSplitter
from rag_agent.embedder import get_embedder
from rag_agent.qdrant_store import get_qdrant_store


class KnowledgeBaseBuilder:
    """
    知识库构建编排器

    负责串联 加载→切分→向量化→入库→标记 全流程。
    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = ParentChildSplitter()
        self.embedder = get_embedder()
        self.qdrant = get_qdrant_store()

    def build(
        self,
        docs_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Dict:
        """
        执行知识库构建。

        Args:
            docs_dir: 文档目录路径（覆盖配置）
            force_rebuild: 是否强制重建（清空已有数据）

        Returns:
            构建统计信息
        """
        if docs_dir:
            self.loader = DocumentLoader(docs_dir)

        print(f"[Builder] 开始构建知识库...")
        print(f"[Builder] 文档目录: {self.loader.docs_dir}")
        print(f"[Builder] 已处理文件数: {self.loader.get_processed_count()}")

        # Step 0: 初始化 Qdrant 集合
        self.qdrant.create_collections(force=force_rebuild)

        # Step 1: 扫描待处理文档
        documents = self.loader.load_all_new()
        if not documents:
            print("[Builder] 没有新文档需要处理")
            return self._summary(0, 0, 0)

        print(f"[Builder] 发现 {len(documents)} 个新文档")

        all_parents = []
        all_children = []

        # Step 2-3: 逐文档切分
        for doc in documents:
            file_name = doc["name"]
            content = doc["content"]
            file_path = doc["path"]

            print(f"[Builder] 处理: {file_name} ({len(content)}字符)")

            try:
                result = self.splitter.split(content, file_name)
                parents = result["parent_chunks"]
                children = result["child_chunks"]

                print(f"  -> 父块: {len(parents)}, 子块: {len(children)}")

                all_parents.extend(parents)
                all_children.extend(children)

                # Step 5: 标记已处理
                self.loader.mark_processed(file_path)

            except Exception as e:
                print(f"  -> 处理失败: {e}")
                continue

        if not all_children:
            print("[Builder] 没有生成有效块")
            return self._summary(len(documents), 0, 0)

        # Step 3-4: 向量化 + 入库 (子块)
        print(f"[Builder] 向量化子块: {len(all_children)}个...")
        child_texts = [c["content"] for c in all_children]
        child_embeddings = self.embedder.encode_documents(child_texts)
        self.qdrant.upsert_children(all_children, child_embeddings)

        # Step 3-4: 向量化 + 入库 (父块)
        print(f"[Builder] 向量化父块: {len(all_parents)}个...")
        parent_texts = [p["content"] for p in all_parents]
        parent_embeddings = self.embedder.encode_documents(parent_texts)
        self.qdrant.upsert_parents(all_parents, parent_embeddings)

        print(f"[Builder] 知识库构建完成")

        return self._summary(len(documents), len(all_parents), len(all_children))

    def _summary(self, files: int, parents: int, children: int) -> Dict:
        """生成构建摘要"""
        info = self.qdrant.get_collection_info()
        return {
            "processed_files": files,
            "parent_chunks": parents,
            "child_chunks": children,
            "total_parents_in_db": info.get(rag_config.parent_collection_name, 0),
            "total_children_in_db": info.get(rag_config.child_collection_name, 0),
            "built_at": datetime.now().isoformat(),
        }

    def get_stats(self) -> Dict:
        """获取知识库统计"""
        info = self.qdrant.get_collection_info()
        return {
            "processed_files": self.loader.get_processed_count(),
            "parent_chunks": info.get(rag_config.parent_collection_name, 0),
            "child_chunks": info.get(rag_config.child_collection_name, 0),
            "documents_dir": str(self.loader.docs_dir),
        }

    def rebuild(self, docs_dir: Optional[str] = None) -> Dict:
        """强制重建知识库"""
        return self.build(docs_dir=docs_dir, force_rebuild=True)

    def update(self, docs_dir: Optional[str] = None) -> Dict:
        """增量更新（仅处理新文档）"""
        return self.build(docs_dir=docs_dir, force_rebuild=False)
