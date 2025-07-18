# 
#引入reasoning teacher，在提出问题对ai要求promt是显示推理链，然后在想后台提出

import os
import json
import requests
import shutil
import datetime
import traceback
import time
import math
from typing import List, Set, Tuple

# 核心依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
# from sentence_transformers.cross_encoder import CrossEncoder # 如果您已安装，可以取消注释

# ======================================================================================
# 0. 自定义的、更稳定的嵌入类
# ======================================================================================
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
        print(f"[自定义嵌入类]: 正在为 {len(texts)} 个文档片段创建嵌入...")
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        print("[自定义嵌入类]: 文档嵌入创建完成。")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding

# ======================================================================================
# 1. 短期工作记忆缓存区 (受LongMem启发)
# ======================================================================================
class ShortTermCache:
    """
    一个模拟LongMem中"Read-Write Memory"的短期工作记忆缓存。
    它维护一个固定容量的近期对话历史，并拥有自己的小型FAISS索引以便快速检索。
    """
    def __init__(self, capacity: int, embeddings_model: StableHuggingFaceEmbeddings):
        self.capacity = capacity
        self.memory: List[Document] = []
        self.embeddings_model = embeddings_model
        self.vector_store: FAISS = None

    def add(self, query: str, response: str):
        """向短期缓存中添加一轮新的对话。如果超出容量，则移除最旧的。"""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0) # FIFO策略，移除最旧的对话

        doc = Document(page_content=f"近期对话: 用户: '{query}', AI助手: '{response}'", metadata={"source": "short_term_cache"})
        self.memory.append(doc)

        if self.memory:
            self.vector_store = FAISS.from_documents(self.memory, self.embeddings_model)
            self.vector_store.embedding_function = self.embeddings_model.embed_query
            print(f"[短期缓存]: 缓存已更新。当前容量: {len(self.memory)}/{self.capacity}")

    def search(self, query: str, k: int) -> List[Document]:
        """从短期缓存中进行相似度搜索。"""
        if not self.vector_store:
            return []
        print(f"[短期缓存]: 正在从短期缓存中检索 {k} 个相关片段...")
        return self.vector_store.similarity_search(query, k=k)

# ======================================================================================
# 2. 配置区域
# ======================================================================================
DEEPSEEK_API_URL = "http://localhost:11434/v1/chat/completions"
DEEPSEEK_API_KEY = "ollama"
DEEPSEEK_MODEL_NAME = "qwen2:7b"

EMBEDDING_MODEL_NAME = "./models/all-MiniLM-L6-v2"
# RERANKER_MODEL_NAME = "./models/ms-marco-MiniLM-L-6-v2" # 如果您已安装，可以取消注释

PROJECT_DIR = "./final_assistant_data/"
MEMORY_FILE = os.path.join(PROJECT_DIR, "memory.json")
FAISS_INDEX_BASE_DIR = os.path.join(PROJECT_DIR, "faiss_indices")

HISTORY_WINDOW_SIZE = 5
SHORT_TERM_CACHE_CAPACITY = 10 

# ======================================================================================
# 3. 核心功能函数
# ======================================================================================

def initialize_memory_file(memory_path: str, user_name: str):
    """初始化内存文件。如果文件不存在，则创建；如果文件存在但用户是新用户，则追加用户信息。"""
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    new_user_data = {
        "name": user_name, "summary": {}, "personality": {},
        "overall_history": "暂无历史摘要。",
        "overall_personality": "用户工作偏好尚不明确。",
        "history": {}
    }
    if not os.path.exists(memory_path):
        print(f"[初始化]: 未找到内存文件，正在为新用户 '{user_name}' 创建: {memory_path}")
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump({user_name: new_user_data}, f, ensure_ascii=False, indent=4)
    else:
        with open(memory_path, "r+", encoding="utf-8") as f:
            try: memory_data = json.load(f)
            except json.JSONDecodeError: memory_data = {}
            if user_name not in memory_data:
                print(f"[初始化]: 内存文件已存在，正在为新用户 '{user_name}' 添加记录...")
                memory_data[user_name] = new_user_data
                f.seek(0)
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
                f.truncate()
            else:
                print(f"[初始化]: 在内存文件中找到现有用户 '{user_name}' 的记录。")

def load_and_index_memory(memory_file_path: str, user_name: str, embeddings_model: StableHuggingFaceEmbeddings, force_rebuild: bool = False):
    """加载或构建长期记忆的FAISS索引。"""
    faiss_index_path = os.path.join(FAISS_INDEX_BASE_DIR, user_name)
    if force_rebuild and os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
    if os.path.exists(faiss_index_path) and not force_rebuild:
        print(f"[系统]: 正在从 '{faiss_index_path}' 为用户 '{user_name}' 加载现有【长期记忆】FAISS索引。")
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        db.embedding_function = embeddings_model.embed_query
        print("[系统]: FAISS 对象的查询函数已修正。")
        return db
    print(f"[系统]: 正在为用户 '{user_name}' 构建新的【长期记忆】FAISS索引...")
    try:
        with open(memory_file_path, 'r', encoding='utf-8') as f:
            user_memory = json.load(f).get(user_name, {})
    except (FileNotFoundError, json.JSONDecodeError): return None
    if not user_memory: return None
    
    docs_to_index = []
    for date, daily_history in user_memory.get('history', {}).items():
        for i, turn in enumerate(daily_history):
            unique_id = f"{date}_{i}"
            docs_to_index.append(Document(
                page_content=f"日期 {date} 的对话: 用户: '{turn['query']}', AI助手: '{turn['response']}'",
                metadata={"source": "history", "unique_id": unique_id}
            ))
    for date, summary in user_memory.get('summary', {}).items():
        docs_to_index.append(Document(page_content=f"日期 {date} 的摘要: {summary.get('content', '')}", metadata={"source": "summary", "unique_id": f"summary_{date}"}))
    docs_to_index.append(Document(page_content=f"全局工作画像: {user_memory.get('overall_personality', '')}", metadata={"source": "overall_personality", "unique_id": "overall_personality"}))
            
    if not docs_to_index:
        db = FAISS.from_documents([Document(page_content=" ")], embeddings_model)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs_to_index)
        db = FAISS.from_documents(split_docs, embeddings_model)
        
    db.embedding_function = embeddings_model.embed_query
    os.makedirs(faiss_index_path, exist_ok=True)
    db.save_local(faiss_index_path)
    print(f"[系统]: 【长期记忆】索引构建完成！已保存至 '{faiss_index_path}'")
    return db

def rerank_documents(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """使用CrossEncoder对检索到的文档进行重排序，以提高相关性。"""
    if not documents: return []
    print(f"[高级技巧]: 正在使用CrossEncoder对 {len(documents)} 个文档进行重排序...")
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        # 懒加载，避免每次都加载模型
        # from sentence_transformers.cross_encoder import CrossEncoder
        # cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
        # 假设您有一个可用的cross_encoder实例
        # scores = cross_encoder.predict(pairs)
        
        # --- 临时替代方案，如果您没有安装CrossEncoder ---
        scores = [1 if query.lower() in doc.page_content.lower() else 0 for doc in documents]
        # --- 临时替代方案结束 ---
        
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"[高级技巧]: 重排序完成，选取Top {top_n} 的结果。")
        return [doc for doc, score in doc_scores[:top_n]]
    except Exception as e:
        print(f"[警告] Re-ranker执行失败: {e}。将返回原始文档。")
        return documents[:top_n]

def answer_question(user_question: str, 
                    long_term_db: FAISS, 
                    short_term_cache: ShortTermCache,
                    history: List[dict], 
                    user_name: str) -> Tuple[str, List[Document]]:
    """
    (受“Reasoning Teachers”启发)
    集成了双层记忆检索和两步式“推理-回答”流程的最终版问答函数。
    """
    
    # --- 步骤 1: 联合检索与重排序 (逻辑不变) ---
    print("[系统]: 正在从【长期记忆】和【短期缓存】中联合检索...")
    long_term_chunks = long_term_db.similarity_search(user_question, k=5)
    short_term_chunks = short_term_cache.search(user_question, k=3)
    
    all_retrieved_chunks = long_term_chunks + short_term_chunks
    unique_docs_map = {doc.page_content: doc for doc in all_retrieved_chunks}
    reranked_chunks = rerank_documents(user_question, list(unique_docs_map.values()), top_n=3)
    
    unique_contents: Set[str] = {chunk.page_content for chunk in reranked_chunks}
    retrieved_context = "\n---\n".join(unique_contents) if unique_contents else "无相关历史资料。"
    
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        professional_profile = json.load(f).get(user_name, {}).get('overall_personality', '用户工作偏好尚不明确。')
    history_prompt = "".join([f"{'用户' if t['role']=='user' else 'AI助手'}: {t['content']}\n" for t in history])

    # --- 步骤 2: 生成推理链 (第一次API调用) ---
    print("[系统]: 正在生成推理链 (CoT)...")
    
    reasoning_prompt = f"""你是一个严谨的逻辑分析师。你的任务是根据用户的【当前问题】和提供的【背景资料】，生成一个解决问题的思考过程。请遵循“假设-验证-结论”或“分步拆解”的逻辑，只输出思考步骤，不要给出最终答案。

### 背景资料:
{retrieved_context}

### 关于用户 {user_name} 的画像:
{professional_profile}

### 近期对话:
{history_prompt}

### 当前问题:
{user_question}

### 你的思考过程:
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    reasoning_payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": reasoning_prompt}], "temperature": 0.5}
    
    try:
        reasoning_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=reasoning_payload, timeout=45)
        reasoning_response.raise_for_status()
        reasoning_chain = reasoning_response.json()['choices'][0]['message']['content']
        print(f"[系统]: 推理链生成成功:\n---\n{reasoning_chain}\n---")
    except Exception as e:
        print(f"[错误] 生成推理链失败: {e}")
        return f"抱歉，在思考如何回答您的问题时遇到了困难: {e}", reranked_chunks

    # --- 步骤 3: 综合推理链生成最终答案 (第二次API调用) ---
    print("[系统]: 正在综合推理链生成最终答案...")

    synthesis_prompt = f"""你是一个高级AI知识助手。你的任务是根据提供的【背景资料】和【AI的思考过程】，为用户的【当前问题】生成一个清晰、简洁、完整的最终答案。

### 背景资料:
{retrieved_context}

### AI的思考过程:
{reasoning_chain}

### 当前问题:
{user_question}

### 你的最终回答:
"""

    synthesis_payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": synthesis_prompt}], "temperature": 0.4}
    
    try:
        synthesis_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=synthesis_payload, timeout=45)
        synthesis_response.raise_for_status()
        final_answer = synthesis_response.json()['choices'][0]['message']['content']
        return final_answer, reranked_chunks
    except Exception as e:
        return f"抱歉，在总结答案时遇到了问题: {e}", reranked_chunks


def reflect_and_update_memory(user_name: str, memory_file_path: str) -> bool:
    """使用“反思”Prompt，进行每日摘要和全局画像更新。"""
    with open(memory_file_path, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)
    user_history = memory_data.get(user_name, {}).get('history', {})
    if not user_history: return False
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if not user_history.get(today_str): return False
    
    conversation_text = "\n".join([f"用户: {t['query']}\nAI助手: {t['response']}" for t in user_history[today_str]])
    summary_prompt = f"""请以客观、精炼的语言，从以下对话中提取关键信息、待办事项(To-do)、或重要决策点。如果无特殊要点，则总结对话的核心议题。

对话内容:
{conversation_text}

今日工作要点总结:"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.1}
    made_update = False
    try:
        daily_summary = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload).json()['choices'][0]['message']['content'].strip()
        if 'summary' not in memory_data[user_name]: memory_data[user_name]['summary'] = {}
        memory_data[user_name]['summary'][today_str] = {"content": daily_summary}
        made_update = True
    except Exception as e: print(f"[错误] 生成每日摘要失败: {e}")
    
    all_summaries = "\n".join([f"日期 {d}: {s.get('content')}" for d, s in memory_data[user_name].get('summary', {}).items()])
    portrait_prompt = f"""请根据以下长期的工作要点摘要，分析并总结用户【{user_name}】的专业领域、研究方向、工作习惯和核心关注点。输出一个简短、客观的第三人称工作画像描述。

工作要点历史:
{all_summaries}

更新后的用户工作画像:"""
    payload['messages'][0]['content'] = portrait_prompt
    try:
        professional_profile = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload).json()['choices'][0]['message']['content'].strip()
        memory_data[user_name]['overall_personality'] = professional_profile
        made_update = True
    except Exception as e: print(f"[错误] 更新全局画像失败: {e}")
    
    if made_update:
        with open(memory_file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
    return made_update


def prune_memory(user_name: str, memory_file: str, retention_threshold: float = 0.25):
    """根据艾宾浩斯遗忘曲线，修剪用户的旧记忆。"""
    print(f"\n[遗忘机制]: 正在为用户 '{user_name}' 检查需要遗忘的记忆...")
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
                    S_factor = 86400  # S因子：1天 = 86400秒，用于减缓遗忘速度
                    S = max(1, strength) * S_factor
                    retention = math.exp(-time_elapsed / S)
                    if retention >= retention_threshold:
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
        print(f"[遗忘机制]: 执行失败，无法读写记忆文件。错误: {e}")

def reinforce_memory(user_name: str, used_chunks: List[Document], memory_file: str):
    """根据用于生成答案的记忆块，加固原始记忆。"""
    if not used_chunks:
        return
    print(f"[记忆加固]: 正在为 {len(used_chunks)} 条被回忆的记忆增强强度...")
    try:
        with open(memory_file, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            if user_name not in memory_data:
                return

            for chunk in used_chunks:
                # 只加固来自长期历史的记忆
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
                            print(f"    - 长期记忆 {unique_id} 强度提升至: {target_turn['strength']}")
                    except (ValueError, IndexError) as e:
                        print(f"    - [警告] 处理记忆ID '{unique_id}' 失败: {e}")
            
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[记忆加固]: 执行失败，无法读写记忆文件。错误: {e}")


# ======================================================================================
# 4. 主程序入口
# ======================================================================================
if __name__ == "__main__":
    try:
        user_name = input("请输入您的名字: ").strip() or "DefaultUser"
        initialize_memory_file(MEMORY_FILE, user_name)
        
        prune_memory(user_name, MEMORY_FILE)
        
        embeddings = StableHuggingFaceEmbeddings(model_name_or_path=EMBEDDING_MODEL_NAME, device='cuda')
        
        # 加载长期记忆
        db = load_and_index_memory(MEMORY_FILE, user_name, embeddings, force_rebuild=True)
        if not db: 
            print(f"[致命错误]: 无法为用户'{user_name}'加载或创建记忆索引。程序退出。")
            exit()
        
        # 初始化短期工作记忆缓存
        short_term_cache = ShortTermCache(capacity=SHORT_TERM_CACHE_CAPACITY, embeddings_model=embeddings)
        
        conversation_history = []
        print(f"\n========================================\n欢迎, {user_name}！我是您的高级AI知识助手(融合最终版)。\n输入 'exit' 或 'quit' 或 '退出' 来结束会话。\n========================================")
        
        while True:
            question = input(f"\n{user_name}: ").strip()
            if question.lower() in ["退出", "exit", "quit"]: break
            if not question: continue
            
            final_answer, used_chunks = answer_question(
                question, 
                db,                  # 长期记忆
                short_term_cache,    # 短期记忆
                conversation_history, 
                user_name
            )
            print(f"\nAI助手: {final_answer}")
            
            # 将当前对话添加到短期缓存
            short_term_cache.add(question, final_answer)

            reinforce_memory(user_name, used_chunks, MEMORY_FILE)
            
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": final_answer})
            if len(conversation_history) > HISTORY_WINDOW_SIZE * 2:
                conversation_history = conversation_history[-HISTORY_WINDOW_SIZE * 2:]
            
            # 写入长期记忆的源文件
            with open(MEMORY_FILE, 'r+', encoding='utf-8') as f:
                memory_data = json.load(f)
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                if 'history' not in memory_data[user_name]: memory_data[user_name]['history'] = {}
                if today_str not in memory_data[user_name]['history']: memory_data[user_name]['history'][today_str] = []
                
                new_turn = {
                    "query": question, 
                    "response": final_answer,
                    "strength": 1,
                    "timestamp": time.time()
                }
                memory_data[user_name]['history'][today_str].append(new_turn)
                
                f.seek(0)
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
                f.truncate()
            
            # 每日反思和重建长期记忆索引
            if reflect_and_update_memory(user_name, MEMORY_FILE):
                print("\n[系统]: 检测到摘要/画像更新，正在重建【长期记忆】知识索引...")
                db = load_and_index_memory(MEMORY_FILE, user_name, embeddings, force_rebuild=True)
        
        print("\nAI助手: 感谢使用，再见！")

    except KeyboardInterrupt:
        print("\n\n程序被手动中断，正在退出...")
    except Exception as e:
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!     程序遇到意外错误      !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        traceback.print_exc()
        print("\n程序已终止。很抱歉出现这个问题。")