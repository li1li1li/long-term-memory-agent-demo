# core_logic.py

import os
import shutil
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 导入配置
import config

# --- 核心功能函数 ---

def build_knowledge_base(force_rebuild=False):
    if force_rebuild and os.path.exists(config.FAISS_INDEX_DIR):
        print("[后台任务：检测到强制重建指令，正在删除旧索引...]")
        shutil.rmtree(config.FAISS_INDEX_DIR)
    if os.path.exists(config.FAISS_INDEX_DIR):
        print(f"[初始化：知识库索引 '{config.FAISS_INDEX_DIR}' 已存在，跳过。]")
        return
    os.makedirs(config.PROJECT_DIR, exist_ok=True)
    if not os.path.exists(config.KNOWLEDGE_BASE_FILE):
        with open(config.KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f: f.write("RAG是检索增强生成的缩写。\n")
    if not os.path.exists(config.MEMORY_BANK_FILE):
        with open(config.MEMORY_BANK_FILE, "w", encoding="utf-8") as f: f.write("这是一个记忆库。\n")
    
    knowledge_text = ""
    with open(config.KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f: knowledge_text += f.read()
    with open(config.MEMORY_BANK_FILE, "r", encoding="utf-8") as f: knowledge_text += "\n" + f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    text_chunks = text_splitter.split_text(knowledge_text)
    
    embeddings_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cuda'})
    db = FAISS.from_texts(text_chunks, embeddings_model)
    db.save_local(config.FAISS_INDEX_DIR)
    print(f"[初始化：知识库索引构建完成！]")

def load_vector_db(embeddings_model):
    if not os.path.exists(config.FAISS_INDEX_DIR): return None
    return FAISS.load_local(config.FAISS_INDEX_DIR, embeddings_model, allow_dangerous_deserialization=True)

def answer_question(user_question, db, history):
    retrieved_chunks = db.similarity_search(user_question, k=3)
    context = "\n---\n".join([chunk.page_content for chunk in retrieved_chunks]) if retrieved_chunks else "无相关信息"
    history_prompt = "".join([f"历史提问: {t['content']}\n" if t['role'] == 'user' else f"历史回答: {t['content']}\n\n" for t in history])
    prompt = f"你是一个专业的问答机器人...【参考资料】:\n{context}\n---\n【历史对话】:\n{history_prompt}\n---\n【当前问题】:\n{user_question}\n【你的回答】:"
    messages_payload = history + [{"role": "user", "content": prompt}]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"}
    payload = {"model": config.DEEPSEEK_MODEL_NAME, "messages": messages_payload, "temperature": 0.3}
    try:
        response = requests.post(config.DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        api_result = response.json()
        choices = api_result.get('choices', [])
        return choices[0].get('message', {}).get('content', '') if choices else "抱歉，AI模型返回了空的回复。"
    except Exception as e:
        return f"抱歉，请求AI服务时遇到了问题: {e}"

def extract_and_save_memory(conversation_turn, existing_memories):
    user_input_escaped = conversation_turn['user'].replace('\\', '\\\\').replace('"', '\\"')
    assistant_output_escaped = conversation_turn['assistant'].replace('\\', '\\\\').replace('"', '\\"')
    extraction_prompt = f"""你是一个信息提取助手...从下面【对话】中提取关于用户的核心事实...如果无，则回答"无"。\n【对话】:\n用户: "{user_input_escaped}"\n机器人: "{assistant_output_escaped}"\n【提取的核心事实】:"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"}
    payload = {"model": config.DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": extraction_prompt}], "temperature": 0.0}
    try:
        response = requests.post(config.DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        api_result = response.json()
        choices = api_result.get('choices', [])
        extracted_facts_text = choices[0].get('message', {}).get('content', '').strip() if choices else ""
        if extracted_facts_text and extracted_facts_text != "无":
            new_facts = [fact.strip() for fact in extracted_facts_text.split('\n') if fact.strip()]
            memory_updated = False
            with open(config.MEMORY_BANK_FILE, "a+", encoding="utf-8") as f:
                for fact in new_facts:
                    if fact not in existing_memories:
                        print(f"[后台任务：发现新记忆 -> '{fact}']")
                        f.write(fact + "\n")
                        existing_memories.add(fact)
                        memory_updated = True
            return memory_updated
        return False
    except Exception as e:
        print(f"[后台任务：记忆提取时发生错误] {e}")
        return False