import os
import json
import shutil
import datetime
import math
import time
import requests
from typing import Tuple, List
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers.cross_encoder import CrossEncoder
# from .state import UserState # 移除顶层导入，以解决循环依赖
from config import (
    MEMORY_FILE, DEEPSEEK_API_URL, DEEPSEEK_API_KEY, 
    DEEPSEEK_MODEL_NAME, RERANKER_MODEL_PATH, FAISS_INDEX_BASE_DIR
)

# --- 初始化模块级资源 ---
print("[系统初始化]: 正在加载Reranker模型...")
try:
    cross_encoder = CrossEncoder(RERANKER_MODEL_PATH)
    print("[系统初始化]: Reranker模型加载成功。")
except Exception as e:
    cross_encoder = None
    print(f"[系统初始化][警告]: Reranker模型加载失败: {e}。将回退到简单排序。")

# --- 核心功能函数 ---

def generate_conversational_response(user_question: str, user_state) -> Tuple[str, List[Document]]:
    """
    (全新核心函数)
    生成一个统一的、兼具高情商和学术能力的对话回复。
    这个函数取代了之前的意图分类和僵化的应答模式。
    """
    # 核心改动：在函数内部导入，打破循环
    from .state import UserState
    assert isinstance(user_state, UserState)

    # 1. 统一检索上下文，为AI提供“思考素材”
    retrieved_context, reranked_chunks = _get_common_context(user_question, user_state)
    
    # 2. 准备近期对话历史
    history_prompt = "".join([f"{'用户' if t['role']=='user' else 'AI助手'}: {t['content']}\n" for t in user_state.conversation_history])

    # 3. 设计全新的“核心人格”Prompt
    persona_prompt = f"""你是一个顶级的AI个人助理，兼具高情商和渊博的学识。

**你的行为准则**:
1.  **主动识别意图**: 自然地判断用户的最新一句话是在【提问】、【陈述事实】还是【闲聊】。
2.  **智能回答**:
    - 如果用户在【提问】，请利用下面提供的【背景资料】来给出全面、清晰的回答。如果资料不足，诚实地说明。
    - 如果用户在【陈述事实】，请用自然、不重复的语言确认你已收到信息（例如“明白了”、“这个信息很有趣”），然后基于这个事实，提出一个相关的、开放式的问题或评论来延续对话，展现你的好奇心。
    - 如果用户在【闲聊】（如打招呼），请用友好、自然的方式回应。
3.  **语言风格**: 你的语言风格是亲切、自信且聪明的。**绝对禁止**使用“好的，收到了”或“好的，我已经记住了”这类僵硬、机械的回复。

**背景资料 (你的记忆)**:
---
{retrieved_context}
---

**近期对话历史**:
{history_prompt}

**用户的最新一句话**:
{user_question}

**你的回复**:
"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": persona_prompt}], "temperature": 0.7}
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        final_answer = response.json()['choices'][0]['message']['content'].strip()
        return final_answer, reranked_chunks
    except Exception as e:
        print(f"[对话回复生成][错误]: {e}")
        return "抱歉，我刚刚好像走神了，你能再说一遍吗？", []


# --- 以下函数保持不变，但会在新的流程中被调用 ---

def initialize_memory_file(memory_path: str, user_name: str):
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    new_user_data = {"name": user_name, "summary": {}, "personality": {}, "history": {}, "facts": []}
    if not os.path.exists(memory_path):
        with open(memory_path, "w", encoding="utf-8") as f: json.dump({user_name: new_user_data}, f, ensure_ascii=False, indent=4)
    else:
        with open(memory_path, "r+", encoding="utf-8") as f:
            try: memory_data = json.load(f)
            except json.JSONDecodeError: memory_data = {}
            if user_name in memory_data and 'facts' not in memory_data[user_name]: memory_data[user_name]['facts'] = []
            if user_name not in memory_data: memory_data[user_name] = new_user_data
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()

def load_and_index_memory(memory_file_path: str, user_name: str, embeddings_model, force_rebuild: bool = False):
    faiss_index_path = os.path.join(FAISS_INDEX_BASE_DIR, user_name)
    if force_rebuild and os.path.exists(faiss_index_path): shutil.rmtree(faiss_index_path)
    if os.path.exists(faiss_index_path) and not force_rebuild:
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        db.embedding_function = embeddings_model.embed_query
        return db
    with open(memory_file_path, 'r', encoding='utf-8') as f: user_memory = json.load(f).get(user_name, {})
    docs_to_index = []
    for date, daily_history in user_memory.get('history', {}).items():
        for i, turn in enumerate(daily_history): docs_to_index.append(Document(page_content=f"日期 {date} 的对话: 用户: '{turn['query']}', AI助手: '{turn['response']}'", metadata={"source": "history", "unique_id": f"{date}_{i}"}))
    for date, summary in user_memory.get('summary', {}).items(): docs_to_index.append(Document(page_content=f"日期 {date} 的摘要: {summary.get('content', '')}", metadata={"source": "summary", "unique_id": f"summary_{date}"}))
    docs_to_index.append(Document(page_content=f"全局工作画像: {user_memory.get('overall_personality', '')}", metadata={"source": "overall_personality", "unique_id": "overall_personality"}))
    for i, fact in enumerate(user_memory.get('facts', [])): docs_to_index.append(Document(page_content=f"用户陈述的一个事实: {fact}", metadata={"source": "fact", "unique_id": f"fact_{i}"}))
    if not docs_to_index: db = FAISS.from_documents([Document(page_content=" ")], embeddings_model)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs_to_index)
        db = FAISS.from_documents(split_docs, embeddings_model)
    db.embedding_function = embeddings_model.embed_query
    os.makedirs(faiss_index_path, exist_ok=True)
    db.save_local(faiss_index_path)
    return db

def rerank_documents(query: str, documents: list, top_n: int = 3) -> list:
    if not documents or not cross_encoder: return documents[:top_n]
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores = cross_encoder.predict(pairs)
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:top_n]]
    except Exception as e:
        print(f"[警告] Re-ranker在预测时出错: {e}")
        return documents[:top_n]

def _get_common_context(user_question: str, user_state) -> Tuple[str, List[Document]]:
    from .state import UserState
    assert isinstance(user_state, UserState)
    long_term_chunks = user_state.long_term_db.similarity_search(user_question, k=5)
    short_term_chunks = user_state.short_term_cache.search(user_question, k=3)
    all_retrieved_chunks = long_term_chunks + short_term_chunks
    unique_docs_map = {doc.page_content: doc for doc in all_retrieved_chunks}
    reranked_chunks = rerank_documents(user_question, list(unique_docs_map.values()), top_n=3)
    retrieved_context = "\n---\n".join([chunk.page_content for chunk in reranked_chunks]) if reranked_chunks else "无相关历史资料。"
    return retrieved_context, reranked_chunks

def answer_question_deep_thought(user_question: str, user_state) -> Tuple[str, List[Document]]:
    from .state import UserState
    assert isinstance(user_state, UserState)
    retrieved_context, reranked_chunks = _get_common_context(user_question, user_state)
    generation_prompt = f"""你是一个富有创造力和逻辑性的思想家。针对用户的【当前问题】和【背景资料】，请从3个不同的、独立互补的角度出发，生成3条解决问题的思考路径或分析大纲。请严格按照以下格式输出，不要添加任何额外解释：\n\n[思考路径 1]\n<这里是第一条思考路径的详细内容>\n\n[思考路径 2]\n<这里是第二条思考路径的详细内容>\n\n[思考路径 3]\n<这里是第三条思考路径的详细内容>\n\n### 背景资料:\n{retrieved_context}\n\n### 当前问题:\n{user_question}"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    llm_payload = {"model": DEEPSEEK_MODEL_NAME, "temperature": 0.7}
    try:
        llm_payload["messages"] = [{"role": "user", "content": generation_prompt}]
        generation_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload, timeout=60)
        generation_response.raise_for_status()
        generated_thoughts_text = generation_response.json()['choices'][0]['message']['content']
        thoughts = {}
        for i in range(1, 4):
            start_tag, end_tag = f"[思考路径 {i}]", f"[思考路径 {i+1}]" if i < 3 else None
            start_index = generated_thoughts_text.find(start_tag)
            if start_index != -1:
                start_index += len(start_tag)
                end_index = generated_thoughts_text.find(end_tag) if end_tag else len(generated_thoughts_text)
                thoughts[f'path_{i}'] = generated_thoughts_text[start_index:end_index].strip()
        if not thoughts: raise ValueError("未能成功解析思考路径。")
    except Exception as e: return f"抱歉，在进行头脑风暴时遇到了问题: {e}", reranked_chunks
    aggregation_input = "".join([f"### 备选思路 {i}:\n{path_text}\n\n" for i, path_text in enumerate(thoughts.values(), 1)])
    aggregation_prompt = f"""你是一位顶级的策略分析师和思想家。你的任务是综合、提炼和升华下面提供的多个【备选思路】，形成一个唯一的、逻辑更严密、内容更全面、结构更清晰的【最终思考方案】。\n\n{aggregation_input}\n---\n### 当前问题:\n{user_question}\n\n### 你的最终思考方案:"""
    try:
        llm_payload["messages"] = [{"role": "user", "content": aggregation_prompt}]
        llm_payload["temperature"] = 0.4
        aggregation_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload, timeout=60)
        aggregation_response.raise_for_status()
        aggregated_thought = aggregation_response.json()['choices'][0]['message']['content']
    except Exception as e: return f"抱歉，在整合思路时遇到了问题: {e}", reranked_chunks
    synthesis_prompt = f"""你是一个高级AI知识助手。你的任务是严格依据下面提供的【最终思考方案】和【原始背景资料】，为用户的【当前问题】生成一个结构清晰、内容详实、措辞专业的最终答案。请直接回答，不要复述思考过程。\n\n### 最终思考方案:\n{aggregated_thought}\n\n### 原始背景资料:\n{retrieved_context}\n\n### 当前问题:\n{user_question}\n\n### 你的最终回答:"""
    try:
        llm_payload["messages"] = [{"role": "user", "content": synthesis_prompt}]
        synthesis_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload, timeout=60)
        synthesis_response.raise_for_status()
        final_answer = synthesis_response.json()['choices'][0]['message']['content']
        return final_answer, reranked_chunks
    except Exception as e: return f"抱歉，在总结最终答案时遇到了问题: {e}", reranked_chunks

def extract_and_store_facts(user_name: str, memory_file: str, user_query: str, ai_response: str) -> bool:
    fact_extraction_prompt = f"""你是一个后台运行的信息提取机器人。请分析以下单轮对话，判断用户是否陈述了任何关于他/她自己的、值得长期记忆的个人信息、偏好或事实。如果对话中不包含任何此类事实，请只回答 "N/A"。\n\n### 对话内容\n用户: "{user_query}"\nAI助手: "{ai_response}"\n\n### 提取出的事实 (请直接列出事实，不要添加任何其他文字):"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": fact_extraction_prompt}], "temperature": 0.0, "stop": ["\n\n"]}
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        extracted_text = response.json()['choices'][0]['message']['content'].strip()
        if extracted_text == "N/A" or not extracted_text: return False
        new_facts = [fact.strip() for fact in extracted_text.split('\n') if fact.strip()]
        if not new_facts: return False
        with open(memory_file, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            if 'facts' not in memory_data[user_name]: memory_data[user_name]['facts'] = []
            for fact in new_facts:
                if fact not in memory_data[user_name]['facts']: memory_data[user_name]['facts'].append(fact)
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
        return True
    except Exception as e:
        print(f"[事实提取][错误]: {e}")
        return False

def reflect_and_update_memory(user_name: str, memory_file_path: str) -> bool:
    try:
        with open(memory_file_path, 'r', encoding='utf-8') as f: memory_data = json.load(f)
        user_history = memory_data.get(user_name, {}).get('history', {})
        if not user_history: return False
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        if not user_history.get(today_str): return False
        if memory_data.get(user_name, {}).get('summary', {}).get(today_str): return False
        print("[反思机制]: 检测到今日新对话，开始进行每日反思...")
        conversation_text = "\n".join([f"用户: {t['query']}\nAI助手: {t['response']}" for t in user_history[today_str]])
        summary_prompt = f"""你是一个专业的会议纪要员和项目助理。请严格按照以下要求，从对话内容中提取结构化的工作信息。\n\n**指令**:\n1. **只关注**与工作、学习、项目、决策相关的实质性内容。\n2. **必须忽略**日常闲聊、情感表达、问候语和与核心工作无关的个人信息。\n3. 如果某个类别下没有内容，请明确写出“无”。\n4. 输出必须严格遵循下面的JSON格式。\n\n**对话内容**:\n{conversation_text}\n\n**输出格式**:\n```json\n{{\n  \"待办事项\": [\n    \"这里列出明确的、需要被执行的待办事项\"\n  ],\n  \"关键决策\": [\n    \"这里列出对话中达成的结论或做出的重要决定\"\n  ],\n  \"核心问题与发现\": [\n    \"这里列出讨论中遇到的主要问题或新的发现\"\n  ]\n}}\n```"""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.0}
        made_update = False
        summary_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        if summary_response.status_code == 200:
            try:
                raw_content = summary_response.json()['choices'][0]['message']['content']
                json_str = raw_content.split('```json')[1].split('```')[0].strip()
                daily_summary_obj = json.loads(json_str)
                daily_summary = (f"待办事项: {'; '.join(daily_summary_obj.get('待办事项', ['无']))}\n"
                                 f"关键决策: {'; '.join(daily_summary_obj.get('关键决策', ['无']))}\n"
                                 f"核心问题与发现: {'; '.join(daily_summary_obj.get('核心问题与发现', ['无']))}")
            except Exception: daily_summary = summary_response.json()['choices'][0]['message']['content'].strip()
            if 'summary' not in memory_data[user_name]: memory_data[user_name]['summary'] = {}
            memory_data[user_name]['summary'][today_str] = {"content": daily_summary}
            made_update = True
            print("[反思机制]: 每日摘要生成成功。")
        else: print(f"[错误] 生成每日摘要失败: {summary_response.text}")
        all_summaries_text = "\n\n".join([f"日期 {d}:\n{s.get('content')}" for d, s in memory_data[user_name].get('summary', {}).items()])
        if all_summaries_text:
            portrait_prompt = f"""你是一位资深的人力资源(HR)和职业规划顾问。请基于以下长期的工作摘要，为用户【{user_name}】提炼出一个专业的、以能力和目标为导向的画像。\n\n**指令**:\n1. 分析用户的核心能力、展现出的技能和知识领域。\n2. 推断用户可能的长期职业目标或当前项目的核心目的。\n3. 识别用户工作模式中潜在的风险或可以改进的地方。\n4. 用简洁、专业的语言总结，分点阐述。\n\n**长期工作摘要历史**:\n{all_summaries_text}\n\n**专业画像总结**:"""
            payload['messages'][0]['content'] = portrait_prompt
            profile_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
            if profile_response.status_code == 200:
                professional_profile = profile_response.json()['choices'][0]['message']['content'].strip()
                memory_data[user_name]['overall_personality'] = professional_profile
                made_update = True
                print("[反思机制]: 全局用户画像更新成功。")
            else: print(f"[错误] 更新全局画像失败: {profile_response.text}")
        if made_update:
            with open(memory_file_path, 'w', encoding='utf-8') as f: json.dump(memory_data, f, ensure_ascii=False, indent=4)
        return made_update
    except Exception as e:
        print(f"[反思机制][错误]: {e}")
        return False

def prune_memory(user_name: str, memory_file: str, retention_threshold: float = 0.25):
    try:
        with open(memory_file, 'r+', encoding='utf-8') as f:
            memory_data = json.load(f)
            if user_name not in memory_data or 'history' not in memory_data[user_name]: return
            current_time = time.time()
            original_turn_count = sum(len(day) for day in memory_data[user_name]['history'].values())
            retained_history = {}
            for date, daily_history in memory_data[user_name]['history'].items():
                retained_turns = []
                for turn in daily_history:
                    time_elapsed = current_time - turn.get("timestamp", current_time)
                    strength = turn.get("strength", 1)
                    S = max(1, strength) * 86400
                    if math.exp(-time_elapsed / S) >= retention_threshold: retained_turns.append(turn)
                if retained_turns: retained_history[date] = retained_turns
            memory_data[user_name]['history'] = retained_history
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
            new_turn_count = sum(len(day) for day in retained_history.values())
            forgotten_count = original_turn_count - new_turn_count
            if forgotten_count > 0: print(f"[遗忘机制]: 完成。共遗忘了 {forgotten_count} 条旧记忆。")
    except (FileNotFoundError, json.JSONDecodeError) as e: print(f"[遗忘机制]: 执行失败: {e}")

def reinforce_memory(user_name: str, used_chunks: List[Document], memory_file: str):
    if not used_chunks: return
    try:
        with open(memory_file, 'r+', encoding='utf-8') as f:
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
                    except (ValueError, IndexError) as e: print(f"    - [警告] 处理记忆ID '{unique_id}' 失败: {e}")
            f.seek(0)
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
            f.truncate()
    except (FileNotFoundError, json.JSONDecodeError) as e: print(f"[记忆加固]: 执行失败: {e}")