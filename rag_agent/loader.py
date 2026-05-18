"""
文档加载器：从固定目录加载文档，支持 MD5 去重校验。

支持格式: .md, .txt, .pdf, .docx
去重策略: MD5 哈希 + 文件路径双重校验
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

from rag_agent.config import rag_config


class DocumentLoader:
    """文档加载器，带 MD5 去重校验"""

    def __init__(self, docs_dir: Optional[str] = None):
        self.docs_dir = Path(docs_dir or rag_config.documents_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.record_file = Path(rag_config.dedup_record_file)
        self.record_file.parent.mkdir(parents=True, exist_ok=True)
        self._processed: Dict[str, dict] = self._load_records()

    def _load_records(self) -> Dict[str, dict]:
        """加载已处理文件记录"""
        if self.record_file.exists():
            try:
                return json.loads(self.record_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                return {}
        return {}

    def _save_records(self):
        """保存处理记录"""
        self.record_file.write_text(
            json.dumps(self._processed, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        """计算文件 MD5 哈希"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_duplicate(self, file_path: Path) -> bool:
        """检查文件是否重复（已处理过且未变化）"""
        abs_path = str(file_path.resolve())
        file_md5 = self._compute_md5(file_path)

        # 按路径检查
        if abs_path in self._processed:
            if self._processed[abs_path].get("md5") == file_md5:
                return True

        # 按 MD5 检查 (不同路径但内容相同)
        for record in self._processed.values():
            if record.get("md5") == file_md5:
                return True

        return False

    def mark_processed(self, file_path: Path):
        """标记文件已处理"""
        abs_path = str(file_path.resolve())
        file_md5 = self._compute_md5(file_path)
        self._processed[abs_path] = {
            "path": abs_path,
            "name": file_path.name,
            "md5": file_md5,
            "size": file_path.stat().st_size,
            "processed_at": datetime.now().isoformat(),
        }
        self._save_records()

    def list_files(self) -> List[Path]:
        """列出待处理文件（排除已处理的）"""
        if not self.docs_dir.exists():
            return []

        supported_exts = {".md", ".txt", ".pdf", ".docx"}
        files = []
        for ext in supported_exts:
            for f in self.docs_dir.rglob(f"*{ext}"):
                files.append(f)
            for f in self.docs_dir.rglob(f"*{ext.upper()}"):
                files.append(f)
        return sorted(set(files))

    def list_new_files(self) -> List[Path]:
        """列出新增或变更的文件（跳过重复）"""
        all_files = self.list_files()
        return [f for f in all_files if not self.is_duplicate(f)]

    def load_content(self, file_path: Path) -> Optional[str]:
        """加载单个文件的内容"""
        ext = file_path.suffix.lower()

        try:
            if ext in [".md", ".txt"]:
                return file_path.read_text(encoding="utf-8")

            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file_path))
                    return "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                except ImportError:
                    raise ImportError("请安装 pypdf: pip install pypdf")

            elif ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(str(file_path))
                    return "\n".join(p.text for p in doc.paragraphs)
                except ImportError:
                    raise ImportError("请安装 python-docx: pip install python-docx")

            else:
                # 尝试作为纯文本
                return file_path.read_text(encoding="utf-8")

        except Exception as e:
            print(f"[Loader] 读取文件失败 {file_path}: {e}")
            return None

    def load_all_new(self) -> List[Dict]:
        """
        加载所有新文档。
        Returns: [{"path": Path, "name": str, "content": str}, ...]
        """
        documents = []
        for file_path in self.list_new_files():
            content = self.load_content(file_path)
            if content:
                documents.append({
                    "path": file_path,
                    "name": file_path.name,
                    "content": content,
                })
        return documents

    def get_processed_count(self) -> int:
        return len(self._processed)
