import os
import shutil
import json
import datetime
import math
import time
from typing import List, Dict, Union
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings # 确保导入 Embeddings 类型提示

from .memory import StableHuggingFaceEmbeddings, ShortTermCache # 假设这两个类在 assistant/memory.py 中
from config import MEMORY_FILE, SHORT_TERM_CACHE_CAPACITY, FAISS_INDEX_BASE_DIR

class UserState:
    """封装单个用户会话的所有状态和资源，并管理记忆的加载、保存和索引。"""

    def __init__(self, user_name: str, embedding_model: StableHuggingFaceEmbeddings):
        print(f"--- 为用户 '{user_name}' 创建/加载新的会话状态 ---")
        self.user_name = user_name
        self.embeddings = embedding_model
        
        # 线性对话历史
        self.conversation_history: List[Dict[str, str]] = []
        # 短期缓存
        self.short_term_cache: ShortTermCache = ShortTermCache(capacity=SHORT_TERM_CACHE_CAPACITY, embeddings_model=self.embeddings)

        # 核心记忆数据
        self.personal_facts: List[Dict] = []
        self.structured_memories: List[Dict] = []

        # [核心修改] 新增一个变更标志，用于追踪内存数据是否发生变化
        self._memory_changed = False
        
        # 初始化时加载数据
        self._ensure_file_structure_and_load_memory()

        # 初始化FAISS向量数据库
        self.long_term_db = self._load_and_index_memory_db(force_rebuild=True)
        
        # 遗忘检查
        self._prune_memory_internal()
        # 遗忘后可能产生变更，统一提交
        self.commit_changes()


    def _mark_memory_as_changed(self):
        """[核心修改] 标记内存数据已发生变化，为后续的统一提交做准备。"""
        if not self._memory_changed:
            print("[UserState]: 内存状态被标记为已更改。")
            self._memory_changed = True

    def _ensure_file_structure_and_load_memory(self):
        """确保记忆文件有正确的 JSON 结构，并处理旧数据迁移，然后将数据加载到内存。"""
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        
        memory_data = {}
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r+", encoding="utf-8") as f:
                try:
                    memory_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[UserState][警告]: 记忆文件 '{MEMORY_FILE}' 损坏，将尝试创建新结构。")
                    memory_data = {}

        if self.user_name not in memory_data:
            memory_data[self.user_name] = {
                "name": self.user_name,
                "personal_facts": [],
                "structured_memories": [],
                "history": {}
            }
        
        user_data_for_migration = memory_data[self.user_name]
        migration_made_changes = False # 追踪迁移是否修改了数据

        user_data_for_migration.setdefault('personal_facts', [])
        user_data_for_migration.setdefault('structured_memories', [])
        user_data_for_migration.setdefault('history', {})

        if 'facts' in user_data_for_migration:
            if user_data_for_migration['facts']:
                migration_made_changes = True
                print(f"[迁移]: 正在迁移用户 {self.user_name} 的旧事实列表到 'personal_facts'...")
                for i, fact_content in enumerate(user_data_for_migration['facts']):
                    if not any(f.get('content') == fact_content for f in user_data_for_migration['personal_facts']):
                        user_data_for_migration['personal_facts'].append({
                            "content": fact_content,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "strength": 1.5,
                            "id": f"personal_fact_legacy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i}"
                        })
            user_data_for_migration.pop('facts', None)
        
        if 'memories' in user_data_for_migration:
            if user_data_for_migration['memories']:
                migration_made_changes = True
                print(f"[迁移]: 正在迁移用户 {self.user_name} 的旧 A-Mem 'memories' 到 'structured_memories'...")
                for mem in user_data_for_migration['memories']:
                    if not any(m.get('id') == mem.get('id') for m in user_data_for_migration['structured_memories']):
                        mem.setdefault('id', f"migrated_mem_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{len(user_data_for_migration['structured_memories'])}")
                        mem.setdefault('type', 'general_note')
                        mem.setdefault('timestamp', datetime.datetime.now().isoformat())
                        mem.setdefault('strength', 1.0)
                        mem.setdefault('links', [])
                        mem.setdefault('contextual_description', mem.get('content', ''))
                        user_data_for_migration['structured_memories'].append(mem)
            user_data_for_migration.pop('memories', None)

        if 'summary' in user_data_for_migration and not any(m.get('type') == 'daily_summary' for m in user_data_for_migration['structured_memories']):
            if user_data_for_migration['summary']:
                migration_made_changes = True
                print(f"[迁移]: 正在迁移用户 {self.user_name} 的旧每日摘要到 'structured_memories'...")
                for date, summary_obj in user_data_for_migration['summary'].items():
                    mem_id = f"daily_summary_{date}"
                    if not any(m['id'] == mem_id for m in user_data_for_migration['structured_memories']):
                        user_data_for_migration['structured_memories'].append({
                            "id": mem_id,
                            "type": "daily_summary",
                            "content": summary_obj.get('content', ''),
                            "timestamp": datetime.datetime.now().isoformat(),
                            "strength": 1.2,
                            "links": [],
                            "contextual_description": f"日期 {date} 的摘要: {summary_obj.get('content','')[:100]}..."
                        })
            user_data_for_migration.pop('summary', None)

        if 'overall_personality' in user_data_for_migration and not any(m.get('type') == 'overall_personality' for m in user_data_for_migration['structured_memories']):
            if user_data_for_migration['overall_personality']:
                migration_made_changes = True
                print(f"[迁移]: 正在迁移用户 {self.user_name} 的旧全局画像到 'structured_memories'...")
                mem_id = f"overall_personality_{self.user_name}"
                user_data_for_migration['structured_memories'].append({
                    "id": mem_id,
                    "type": "overall_personality",
                    "content": user_data_for_migration['overall_personality'],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "strength": 2.0,
                    "links": [],
                    "contextual_description": f"用户 {self.user_name} 的全局工作画像。"
                })
            user_data_for_migration.pop('overall_personality', None)

        self.personal_facts = user_data_for_migration.get('personal_facts', [])
        self.structured_memories = user_data_for_migration.get('structured_memories', [])
        
        linear_history = []
        for date, turns in user_data_for_migration.get('history', {}).items():
            for turn in turns:
                timestamp_str = turn.get('timestamp', datetime.datetime.now().isoformat())
                try:
                    datetime.datetime.fromisoformat(timestamp_str)
                except ValueError:
                    timestamp_str = datetime.datetime.now().isoformat()
                linear_history.append({
                    "role": turn.get('role', 'user' if 'query' in turn else 'assistant'),
                    "content": turn.get('query', turn.get('response', turn.get('content', ''))),
                    "timestamp": timestamp_str
                })
        self.conversation_history = sorted(linear_history, key=lambda x: x['timestamp'])

        for i, fact in enumerate(self.personal_facts):
            fact.setdefault('id', f"personal_fact_gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i}")
            fact.setdefault('timestamp', datetime.datetime.now().isoformat())
            fact.setdefault('strength', 1.0)
        
        if migration_made_changes:
             with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
        print(f"[UserState]: 用户 '{self.user_name}' 的记忆文件结构已确保并数据加载完成。")


    def _load_and_index_memory_db(self, force_rebuild: bool = False):
        """加载并索引所有长期记忆到 FAISS。"""
        faiss_index_path = os.path.join(FAISS_INDEX_BASE_DIR, self.user_name)
        if force_rebuild and os.path.exists(faiss_index_path):
            print(f"[FAISS]: 强制重建，删除旧索引：{faiss_index_path}")
            shutil.rmtree(faiss_index_path)
        
        if os.path.exists(faiss_index_path) and not force_rebuild:
            try:
                db = FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
                db.embedding_function = self.embeddings.embed_query
                print(f"[FAISS]: 已从本地加载 {self.user_name} 的向量数据库。")
                return db
            except Exception as e:
                print(f"[FAISS][错误]: 无法加载本地向量数据库: {e}。将重建。")

        docs_to_index = []

        for fact in self.personal_facts:
            docs_to_index.append(Document(
                page_content=fact.get('content'),
                metadata={
                    "source": "personal_facts",
                    "id": fact.get('id'),
                    "type": "user_profile_fact",
                    "strength": fact.get('strength', 1.0)
                }
            ))
        
        for mem in self.structured_memories:
            content_for_index = mem.get('contextual_description', mem.get('content', ''))
            mem_type = mem.get('type', 'general_note')
            
            docs_to_index.append(Document(
                page_content=content_for_index,
                metadata={"source": "agentic_memory_structured", "id": mem['id'], "type": mem_type, "strength": mem.get('strength', 1.0)}
            ))

        if not docs_to_index:
            db = FAISS.from_documents([Document(page_content="初始记忆库为空。")], self.embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs_to_index)
            db = FAISS.from_documents(split_docs, self.embeddings)

        db.embedding_function = self.embeddings.embed_query
        os.makedirs(faiss_index_path, exist_ok=True)
        db.save_local(faiss_index_path)
        print(f"[FAISS]: 已为 {self.user_name} 重建并保存向量数据库。")
        return db

    def save_all_memory_to_file(self):
        """将当前 UserState 实例中的所有内存数据保存到 JSON 文件。"""
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            memory_data = {}
        
        if self.user_name not in memory_data:
            memory_data[self.user_name] = {}
        
        memory_data[self.user_name]['name'] = self.user_name
        memory_data[self.user_name]['personal_facts'] = self.personal_facts
        memory_data[self.user_name]['structured_memories'] = self.structured_memories
        
        history_by_date = {}
        for turn in self.conversation_history:
            try:
                dt_obj = datetime.datetime.fromisoformat(turn.get('timestamp', datetime.datetime.now().isoformat()))
                date_str = dt_obj.strftime("%Y-%m-%d")
            except ValueError:
                date_str = datetime.date.today().strftime("%Y-%m-%d")
            
            if date_str not in history_by_date:
                history_by_date[date_str] = []
            
            history_entry = {"role": turn.get('role', 'unknown'), "content": turn.get('content', ''), "timestamp": turn.get('timestamp')}
            history_by_date[date_str].append(history_entry)
        
        memory_data[self.user_name]['history'] = history_by_date
        
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
        print(f"[UserState]: 用户 '{self.user_name}' 状态已保存到文件。")

    def commit_changes(self):
        """
        [核心修改] 统一的提交方法：如果内存被标记为已更改，
        则一次性保存到文件并重建索引。
        """
        if self._memory_changed:
            print("[UserState]: 检测到内存变更，正在统一保存并重建索引...")
            self.save_all_memory_to_file()
            self.long_term_db = self._load_and_index_memory_db(force_rebuild=True)
            self._memory_changed = False # 重置标志
            print("[UserState]: 统一更新完成。")
        else:
            print("[UserState]: 无内存变更，跳过保存和索引重建。")

    # --- [核心修改] 新增高级内存操作方法，供logic.py调用 ---
    
    def add_personal_fact(self, fact_content: str) -> bool:
        """向内存中添加个人事实，并标记变更。"""
        if not any(f['content'] == fact_content for f in self.personal_facts):
            new_fact = {
                "content": fact_content,
                "timestamp": datetime.datetime.now().isoformat(),
                "strength": 1.5,
                "id": f"personal_fact_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            }
            self.personal_facts.append(new_fact)
            self._mark_memory_as_changed()
            return True
        return False

    def add_or_update_structured_memory(self, memory_obj: Dict):
        """向内存中添加或更新结构化记忆，并标记变更。"""
        is_update = False
        if memory_obj['type'] in ["overall_personality", "daily_summary"]:
            for i, mem in enumerate(self.structured_memories):
                if mem['id'] == memory_obj['id']:
                    self.structured_memories[i] = memory_obj
                    is_update = True
                    break
        
        if not is_update:
            self.structured_memories.append(memory_obj)

        # 更新反向链接
        for linked_id in memory_obj.get("links", []):
            for mem in self.structured_memories:
                if mem['id'] == linked_id and memory_obj['id'] not in mem.get('links', []):
                    mem.setdefault('links', []).append(memory_obj['id'])
                    break
        
        self._mark_memory_as_changed()
        print(f"[UserState]: {'更新' if is_update else '添加'}了结构化记忆 (ID: {memory_obj['id']}) 至内存。")

    def reinforce_memory_in_state(self, used_chunks: List[Document]):
        """在内存中增强记忆强度，并标记变更。"""
        if not used_chunks: return
        updated = False
        for chunk in used_chunks:
            mem_id = chunk.metadata.get("id")
            if not mem_id: continue

            target_list = None
            if chunk.metadata.get("type") == "user_profile_fact":
                target_list = self.personal_facts
            elif chunk.metadata.get("source") == "agentic_memory_structured":
                target_list = self.structured_memories
            
            if target_list:
                for item in target_list:
                    if item.get('id') == mem_id:
                        item['strength'] = item.get('strength', 1.0) + 0.1
                        item['timestamp'] = datetime.datetime.now().isoformat()
                        updated = True
                        print(f"    - 记忆 {mem_id} 强度提升至: {item['strength']:.2f}")
                        break
        if updated:
            self._mark_memory_as_changed()
    
    def _prune_memory_internal(self, retention_threshold: float = 0.25):
        """内部调用，根据遗忘曲线修剪内存中的记忆。"""
        print(f"[UserState][遗忘机制]: 正在为用户 '{self.user_name}' 检查需要遗忘的记忆...")
        current_time = time.time()
        updated = False

        original_personal_facts_count = len(self.personal_facts)
        retained_personal_facts = []
        for fact in self.personal_facts:
            fact_timestamp = datetime.datetime.fromisoformat(fact.get("timestamp", datetime.datetime.now().isoformat())).timestamp()
            time_elapsed = current_time - fact_timestamp
            strength = fact.get("strength", 1.0)
            S = max(1.0, strength) * (86400 * 30)
            retention = math.exp(-time_elapsed / S)
            if retention >= retention_threshold or strength >= 5.0:
                retained_personal_facts.append(fact)
            else:
                print(f"    - 遗忘个人事实: {fact.get('content', '')[:50]}... (ID: {fact['id']}, 强度: {strength:.2f}, 保留率: {retention:.2f})")
                updated = True
        
        if len(retained_personal_facts) != original_personal_facts_count:
            self.personal_facts = retained_personal_facts
            forgotten_facts_count = original_personal_facts_count - len(retained_personal_facts)
            print(f"[UserState][遗忘机制]: 遗忘了 {forgotten_facts_count} 条个人事实。")

        original_structured_mems_count = len(self.structured_memories)
        retained_structured_memories = []
        for mem in self.structured_memories:
            if mem.get('type') == 'overall_personality':
                retained_structured_memories.append(mem)
                continue
            mem_timestamp = datetime.datetime.fromisoformat(mem.get("timestamp", datetime.datetime.now().isoformat())).timestamp()
            time_elapsed = current_time - mem_timestamp
            strength = mem.get("strength", 1.0)
            S = max(1.0, strength) * (86400 * 7)
            retention = math.exp(-time_elapsed / S)
            if retention >= retention_threshold:
                retained_structured_memories.append(mem)
            else:
                print(f"    - 遗忘结构化记忆: {mem.get('contextual_description', mem.get('content'))[:50]}... (ID: {mem['id']}, 类型: {mem.get('type')}, 强度: {strength:.2f}, 保留率: {retention:.2f})")
                updated = True
        
        if len(retained_structured_memories) != original_structured_mems_count:
            self.structured_memories = retained_structured_memories
            forgotten_structured_mems_count = original_structured_mems_count - len(retained_structured_memories)
            print(f"[UserState][遗忘机制]: 遗忘了 {forgotten_structured_mems_count} 条结构化记忆。")

        if updated:
            self._mark_memory_as_changed()
        else:
            print("[UserState][遗忘机制]: 没有需要遗忘的记忆。")

    def clear_memory_and_restart(self):
        """清除当前用户的记忆数据并重置状态。"""
        print(f"--- 清除用户 '{self.user_name}' 的记忆并重置 ---")
        try:
            with open(MEMORY_FILE, 'r+', encoding='utf-8') as f:
                memory_data = json.load(f)
                if self.user_name in memory_data:
                    del memory_data[self.user_name]
                f.seek(0)
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
                f.truncate()
            print(f"[UserState]: 用户 '{self.user_name}' 的文件记忆已清除。")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[UserState][错误]: 清除文件记忆失败: {e}")

        faiss_index_path = os.path.join(FAISS_INDEX_BASE_DIR, self.user_name)
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)
            print(f"[UserState]: 用户 '{self.user_name}' 的FAISS索引已清除。")

        self.personal_facts = []
        self.structured_memories = []
        self.conversation_history = []
        self.short_term_cache = ShortTermCache(capacity=SHORT_TERM_CACHE_CAPACITY, embeddings_model=self.embeddings)
        self.long_term_db = self._load_and_index_memory_db(force_rebuild=True)
        self._memory_changed = False # 确保标志也被重置
        print(f"--- 用户 '{self.user_name}' 的会话状态已重置 ---")