# embedding_utils.py

from sentence_transformers import SentenceTransformer
from typing import List

class StableHuggingFaceEmbeddings:
    """一个更稳定的自定义嵌入类，它直接、简单地使用 sentence-transformers 库的核心功能。"""
    def __init__(self, model_name_or_path: str, device: str = 'cuda'):
        try:
            print(f"[嵌入工具]: 正在从 '{model_name_or_path}' 加载嵌入模型...")
            self.model = SentenceTransformer(model_name_or_path, device=device)
            print("[嵌入工具]: 嵌入模型加载成功。")
        except Exception as e:
            print(f"[嵌入工具]: 致命错误：加载SentenceTransformer模型失败。错误信息: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"[嵌入工具]: 正在为 {len(texts)} 个文档片段创建嵌入...")
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        print("[嵌入工具]: 文档嵌入创建完成。")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding