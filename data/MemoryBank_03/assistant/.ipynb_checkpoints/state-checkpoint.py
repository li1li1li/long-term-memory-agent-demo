# assistant/state.py
import os
import shutil
from .memory import StableHuggingFaceEmbeddings, ShortTermCache
from .logic import initialize_memory_file, prune_memory, load_and_index_memory
from config import MEMORY_FILE, SHORT_TERM_CACHE_CAPACITY

class UserState:
    """封装单个用户会话的所有状态和资源。"""
    def __init__(self, user_name: str, embedding_model: StableHuggingFaceEmbeddings):
        print(f"--- 为用户 '{user_name}' 创建新的会话状态 ---")
        self.user_name = user_name
        self.embeddings = embedding_model
        
        initialize_memory_file(MEMORY_FILE, self.user_name)
        prune_memory(self.user_name, MEMORY_FILE)
        
        self.long_term_db = load_and_index_memory(MEMORY_FILE, self.user_name, self.embeddings, force_rebuild=True)
        self.short_term_cache = ShortTermCache(capacity=SHORT_TERM_CACHE_CAPACITY, embeddings_model=self.embeddings)
        self.conversation_history = []

    def clear_memory_and_restart(self):
        """清除该用户的记忆并重新初始化。"""
        print(f"--- 清除用户 '{self.user_name}' 的记忆并重启 ---")
        faiss_dir = f'./final_assistant_data/faiss_indices/{self.user_name}'
        if os.path.exists(faiss_dir):
            shutil.rmtree(faiss_dir)
        self.__init__(self.user_name, self.embeddings)