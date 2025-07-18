# memory_manager.py

import os
import json
import shutil
import time
import datetime
import math
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from embedding_utils import StableHuggingFaceEmbeddings
import config

def initialize_memory_file(user_name: str):
    """初始化指定用户的内存文件。"""
    os.makedirs(os.path.dirname(config.MEMORY_FILE), exist_ok=True)
    new_user_data = {
        "name": user_name, "summary": {}, "personality": {},
        "overall_history": "暂无历史摘要。",
        "overall_personality": "用户工作偏好尚不明确。",
        "history": {}
    }
    if not os.path.exists(config.MEMORY_FILE):
        print(f"[记忆管理]: 未找到内存文件，正在为新用户 '{user_name}' 创建...")
        with open(config.MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({user_name: new_user_data}, f, ensure_ascii=False, indent=4)
    else:
        with open(config.MEMORY_FILE, "r+", encoding="utf-8") as f:
            try: memory_data = json.load(f)
            except json.JSONDecodeError: memory_data = {}
            if user_name not in memory_data:
                print(f"[记忆管理]: 内存文件已存在，正在为新用户 '{user_name}' 添加记录...")
                memory_data[user_name] = new_user_data
                f.seek(0)
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
                f.truncate()
            else:
                print(f"[记忆管理]: 在内存文件中找到现有用户 '{user_name}' 的记录。")

def load_and_index_memory(user_name: str, embeddings_model: StableHuggingFaceEmbeddings, force_rebuild: bool = False):
    """为指定用户加载或构建FAISS索引。"""
    faiss_index_path = os.path.join(config.FAISS_INDEX_BASE_DIR, user_name)
    if force_rebuild and os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
    if os.path.exists(faiss_index_path) and not force_rebuild:
        print(f"[记忆管理]: 正在从 '{faiss_index_path}' 为用户 '{user_name}' 加载现有FAISS索引。")
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        db.embedding_function = embeddings_model.embed_query
        return db
    print(f"[记忆管理]: 正在为用户 '{user_name}' 构建新的FAISS索引...")
    try:
        with open(config.MEMORY_FILE, 'r', encoding='utf-8') as f:
            user_memory = json.load(f).get(user_name, {})
    except (FileNotFoundError, json.JSONDecodeError): return None
    if not user_memory: return None
    docs_to_index = []
    for date, daily_history in user_memory.get('history', {}).items():
        for i, turn in enumerate(daily_history):
            unique_id = f"{date}_{i}"
            docs_to_index.append(Document(page_content=f"日期 {date} 的对话: 用户: '{turn['query']}', AI助手: '{turn['response']}'", metadata={"source": "history", "unique_id": unique_id}))
    if not docs_to_index:
        db = FAISS.from_documents([Document(page_content=" ")], embeddings_model)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs_to_index)
        db = FAISS.from_documents(split_docs, embeddings_model)
    db.embedding_function = embeddings_model.embed_query
    os.makedirs(faiss_index_path, exist_ok=True)
    db.save_local(faiss_index_path)
    print(f"[记忆管理]: 索引构建完成！已保存至 '{faiss_index_path}'")
    return db

def prune_memory(user_name: str):
    """根据艾宾浩斯遗忘曲线，修剪用户的旧记忆。"""
    print(f"\n[记忆管理]: 正在为用户 '{user_name}' 执行遗忘检查...")
    try:
        with open(config.MEMORY_FILE, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            if user_name not in memory_data or 'history' not in memory_data[user_name]: return
            current_time = time.time()
            original_turn_count = sum(len(day) for day in memory_data[user_name]['history'].values())
            retained_history = {}
            for date, daily_history in memory_data[user_name]['history'].items():
                retained_turns = [turn for turn in daily_history if math.exp(-(current_time - turn.get("timestamp", current_time)) / (max(1, turn.get("strength", 1)) * 86400)) >= 0.25]
                if retained_turns: retained_history[date] = retained_turns
            memory_data[user_name]['history'] = retained_history
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
            forgotten_count = original_turn_count - sum(len(day) for day in retained_history.values())
            if forgotten_count > 0: print(f"[记忆管理]: 遗忘检查完成，遗忘了 {forgotten_count} 条旧记忆。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[记忆管理]: 遗忘检查失败，错误: {e}")

def reinforce_memory(user_name: str, used_chunks: List[Document]):
    """加固被回忆的记忆。"""
    if not used_chunks: return
    print(f"[记忆管理]: 正在为 {len(used_chunks)} 条被回忆的记忆增强强度...")
    try:
        with open(config.MEMORY_FILE, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            if user_name not in memory_data: return
            for chunk in used_chunks:
                if chunk.metadata.get("source") == "history":
                    unique_id = chunk.metadata.get("unique_id")
                    if not unique_id: continue
                    try:
                        date, turn_index_str = unique_id.split('_')
                        turn_index = int(turn_index_str)
                        if date in memory_data[user_name]['history'] and len(memory_data[user_name]['history'][date]) > turn_index:
                            target_turn = memory_data[user_name]['history'][date][turn_index]
                            target_turn['strength'] = target_turn.get('strength', 1) + 1
                            target_turn['timestamp'] = time.time()
                            print(f"    - 记忆 {unique_id} 强度提升至: {target_turn['strength']}")
                    except (ValueError, IndexError) as e:
                        print(f"    - [警告] 处理记忆ID '{unique_id}' 失败: {e}")
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[记忆管理]: 记忆加固失败，错误: {e}")

def save_conversation_turn(user_name, question, final_answer):
    """将一轮新的对话保存到记忆文件中。"""
    try:
        with open(config.MEMORY_FILE, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            if 'history' not in memory_data[user_name]: memory_data[user_name]['history'] = {}
            if today_str not in memory_data[user_name]['history']: memory_data[user_name]['history'][today_str] = []
            new_turn = {"query": question, "response": final_answer, "strength": 1, "timestamp": time.time()}
            memory_data[user_name]['history'][today_str].append(new_turn)
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
    except (FileNotFoundError, json.JSONDecodeError) as e:
         print(f"[记忆管理]: 保存对话失败，错误: {e}")