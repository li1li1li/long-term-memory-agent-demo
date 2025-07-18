# assistant/state.py (修正版)

import os
import shutil
import json
import datetime
import math
import time

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .memory import StableHuggingFaceEmbeddings, ShortTermCache
from config import MEMORY_FILE, SHORT_TERM_CACHE_CAPACITY, FAISS_INDEX_BASE_DIR

# --- 以下三个函数从 logic.py 移至此处 ---

def initialize_memory_file(memory_path: str, user_name: str):
    """初始化内存文件，并确保 'facts' 列表存在。"""
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    new_user_data = {
        "name": user_name, "summary": {}, "personality": {}, 
        "history": {}, "facts": []
    }
    if not os.path.exists(memory_path):
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump({user_name: new_user_data}, f, ensure_ascii=False, indent=4)
    else:
        with open(memory_path, "r+", encoding="utf-8") as f:
            try:
                memory_data = json.load(f)
            except json.JSONDecodeError:
                memory_data = {}
            if user_name in memory_data and 'facts' not in memory_data[user_name]:
                memory_data[user_name]['facts'] = []
            if user_name not in memory_data:
                memory_data[user_name] = new_user_data
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()

def prune_memory(user_name: str, memory_file: str, retention_threshold: float = 0.25):
    """根据艾宾浩斯遗忘曲线，修剪用户的旧记忆。"""
    try:
        with open(memory_file, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            if user_name not in memory_data or 'history' not in memory_data[user_name]:
                return
            
            current_time = time.time()
            original_turn_count = sum(len(day) for day in memory_data[user_name]['history'].values())
            
            retained_history = {}
            for date, daily_history in memory_data[user_name]['history'].items():
                retained_turns = []
                for turn in daily_history:
                    time_elapsed = current_time - turn.get("timestamp", current_time)
                    strength = turn.get("strength", 1)
                    S = max(1, strength) * 86400
                    if math.exp(-time_elapsed / S) >= retention_threshold:
                        retained_turns.append(turn)
                if retained_turns:
                    retained_history[date] = retained_turns
            
            memory_data[user_name]['history'] = retained_history
            
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
            
            new_turn_count = sum(len(day) for day in retained_history.values())
            forgotten_count = original_turn_count - new_turn_count
            if forgotten_count > 0:
                print(f"[遗忘机制]: 完成。共遗忘了 {forgotten_count} 条旧记忆。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[遗忘机制]: 执行失败: {e}")

def load_and_index_memory(memory_file_path: str, user_name: str, embeddings_model, force_rebuild: bool = False):
    """加载或构建长期记忆的FAISS索引，包含所有类型的记忆。"""
    faiss_index_path = os.path.join(FAISS_INDEX_BASE_DIR, user_name)
    if force_rebuild and os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
    if os.path.exists(faiss_index_path) and not force_rebuild:
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        db.embedding_function = embeddings_model.embed_query
        return db
    
    with open(memory_file_path, 'r', encoding='utf-8') as f:
        user_memory = json.load(f).get(user_name, {})
    
    docs_to_index = []
    for date, daily_history in user_memory.get('history', {}).items():
        for i, turn in enumerate(daily_history):
            docs_to_index.append(Document(page_content=f"日期 {date} 的对话: 用户: '{turn['query']}', AI助手: '{turn['response']}'", metadata={"source": "history", "unique_id": f"{date}_{i}"}))
    for date, summary in user_memory.get('summary', {}).items():
        docs_to_index.append(Document(page_content=f"日期 {date} 的摘要: {summary.get('content', '')}", metadata={"source": "summary", "unique_id": f"summary_{date}"}))
    docs_to_index.append(Document(page_content=f"全局工作画像: {user_memory.get('overall_personality', '')}", metadata={"source": "overall_personality", "unique_id": "overall_personality"}))
    for i, fact in enumerate(user_memory.get('facts', [])):
        docs_to_index.append(Document(page_content=f"用户陈述的一个事实: {fact}", metadata={"source": "fact", "unique_id": f"fact_{i}"}))
    
    if not docs_to_index:
        db = FAISS.from_documents([Document(page_content=" ")], embeddings_model)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs_to_index)
        db = FAISS.from_documents(split_docs, embeddings_model)
        
    db.embedding_function = embeddings_model.embed_query
    os.makedirs(faiss_index_path, exist_ok=True)
    db.save_local(faiss_index_path)
    return db


class UserState:
    """封装单个用户会话的所有状态和资源。"""
    def __init__(self, user_name: str, embedding_model: StableHuggingFaceEmbeddings):
        print(f"--- 为用户 '{user_name}' 创建新的会话状态 ---")
        self.user_name = user_name
        self.embeddings = embedding_model
        
        # 现在直接调用本文件内的函数
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
        # 重新初始化会话状态
        self.__init__(self.user_name, self.embeddings)