import os
# 必须放在导入 HuggingFaceBgeEmbeddings 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class EmbeddingClientManager:
    def __init__(self):
        self.client: HuggingFaceBgeEmbeddings | None = None

    def init(self):
        self.client: HuggingFaceBgeEmbeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )


embedding_client_manager = EmbeddingClientManager()
