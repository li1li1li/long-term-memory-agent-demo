# assistant/memory.py
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from typing import List

class StableHuggingFaceEmbeddings:
    """一个更稳定的自定义嵌入类，它直接、简单地使用 sentence-transformers 库的核心功能。"""
    def __init__(self, model_name_or_path: str, device: str = 'cuda'):
        try:
            print(f"[自定义嵌入类]: 正在从 '{model_name_or_path}' 加载嵌入模型...")
            self.model = SentenceTransformer(model_name_or_path, device=device)
            print("[自定义嵌入类]: 嵌入模型加载成功。")
        except Exception as e:
            print(f"[自定义嵌入类]: 致命错误：加载SentenceTransformer模型失败。错误信息: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding

class ShortTermCache:
    """一个模拟LongMem中"Read-Write Memory"的短期工作记忆缓存。"""
    def __init__(self, capacity: int, embeddings_model: StableHuggingFaceEmbeddings):
        self.capacity = capacity
        self.memory: List[Document] = []
        self.embeddings_model = embeddings_model
        self.vector_store: FAISS = None

    def add(self, query: str, response: str):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        doc = Document(page_content=f"近期对话: 用户: '{query}', AI助手: '{response}'", metadata={"source": "short_term_cache"})
        self.memory.append(doc)

        if self.memory:
            self.vector_store = FAISS.from_documents(self.memory, self.embeddings_model)
            self.vector_store.embedding_function = self.embeddings_model.embed_query

    def search(self, query: str, k: int) -> List[Document]:
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)