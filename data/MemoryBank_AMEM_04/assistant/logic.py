import os
import json
import requests
import datetime
from typing import Tuple, List, Dict
from langchain.docstore.document import Document
from sentence_transformers.cross_encoder import CrossEncoder

from config import (
    DEEPSEEK_API_URL, DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL_NAME, RERANKER_MODEL_PATH
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
def generate_conversational_response(user_question: str, user_state, deep_mode: bool) -> Tuple[str, List[Document]]:
    """生成对话回复。"""
    from .state import UserState 
    assert isinstance(user_state, UserState)

    if deep_mode:
        print("[AI助手模式]: 启动深度思考模式。")
        return _execute_deep_thought_mode(user_question, user_state)
    else:
        print("[AI助手模式]: 启动标准对话模式。")
        return _execute_standard_mode(user_question, user_state)

def _execute_standard_mode(user_question: str, user_state) -> Tuple[str, List[Document]]:
    retrieved_context, reranked_chunks = _get_common_context(user_question, user_state)
    history_prompt = "".join([f"{'用户' if t['role']=='user' else 'AI助手'}: {t['content']}\n" for t in user_state.conversation_history])

    persona_prompt = f"""你是一个顶级的AI个人助理，兼具高情商和渊博的学识。

**你的行为准则**:
1.  **主动识别意图**: 自然地判断用户的最新一句话是在【提问】、【陈述事实】还是【闲聊】。
2.  **智能回答**:
    - 如果用户在【提问】，请利用下面提供的【背景资料】来给出全面、清晰的回答。如果资料不足，诚实地说明。
    - 如果用户在【陈述事实】，请用自然、不重复的语言确认你已收到信息（例如“明白了”、“这个信息很有趣”），然后基于这个事实，提出一个相关的、开放式的问题或评论来延续对话，展现你的好奇心。
    - 如果用户在【闲聊】（如打招呼），请用友好、自然的方式回应。
3.  **语言风格**: 你的语言风格是亲切、自信且聪明的。**绝对禁止**使用“好的，收到了”或“好的，我已经记住了”这类僵硬、机械的回复。

**背景资料 (你的长期记忆)**:
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
        print(f"[标准模式][错误]: {e}")
        return "抱歉，标准模式下回答问题时遇到错误。", []


def _execute_deep_thought_mode(user_question: str, user_state) -> Tuple[str, List[Document]]:
    retrieved_context, reranked_chunks = _get_common_context(user_question, user_state)
    history_prompt = "".join([f"{'用户' if t['role']=='user' else 'AI助手'}: {t['content']}\n" for t in user_state.conversation_history])

    # 1. 思想生成阶段
    generation_prompt = f"""你是一个富有创造力和逻辑性的思想家。针对用户的【当前问题】和提供的【背景资料】，请从3个不同的、独立互补的角度出发，生成3条解决问题的思考路径或分析大纲。请严格按照以下格式输出，不要添加任何额外解释：

### 背景资料 (包括检索的长期记忆和近期对话历史):
{retrieved_context}
{history_prompt}
---
### 当前问题:
{user_question}

---
[思考路径 1]
<这里是第一条思考路径的详细内容>

[思考路径 2]
<这里是第二条思考路径的详细内容>

[思考路径 3]
<这里是第三条思考路径的详细内容>
"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    llm_payload = {"model": DEEPSEEK_MODEL_NAME, "temperature": 0.7} 
    thoughts = {}
    try:
        llm_payload["messages"] = [{"role": "user", "content": generation_prompt}]
        generation_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload, timeout=60)
        generation_response.raise_for_status()
        generated_thoughts_text = generation_response.json()['choices'][0]['message']['content']
        for i in range(1, 4):
            start_tag, end_tag = f"[思考路径 {i}]", f"[思考路径 {i+1}]" if i < 3 else None
            start_index = generated_thoughts_text.find(start_tag)
            if start_index != -1:
                start_index += len(start_tag)
                end_index = generated_thoughts_text.find(end_tag) if end_tag else len(generated_thoughts_text)
                thoughts[f'path_{i}'] = generated_thoughts_text[start_index:end_index].strip()
        if not thoughts: raise ValueError("未能成功解析思考路径。")
    except Exception as e: return f"抱歉，在进行头脑风暴时遇到了问题: {e}", reranked_chunks

    # 2. 思想聚合阶段
    aggregation_input = "".join([f"### 备选思路 {i}:\n{path_text}\n\n" for i, path_text in enumerate(thoughts.values(), 1)])
    aggregation_prompt = f"""你是一位顶级的策略分析师和思想家。你的任务是综合、提炼和升华下面提供的多个【备选思路】，形成一个唯一的、逻辑更严密、内容更全面、结构更清晰的【最终思考方案】。\n\n{aggregation_input}\n---\n### 当前问题:\n{user_question}\n\n### 你的最终思考方案:"""
    try:
        llm_payload["messages"] = [{"role": "user", "content": aggregation_prompt}]
        llm_payload["temperature"] = 0.4 
        aggregation_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload, timeout=60)
        aggregation_response.raise_for_status()
        aggregated_thought = aggregation_response.json()['choices'][0]['message']['content']
    except Exception as e: return f"抱歉，在整合思路时遇到了问题: {e}", reranked_chunks

    # 3. 最终答案生成阶段
    synthesis_prompt = f"""你是一个高级AI知识助手。你的任务是严格依据下面提供的【最终思考方案】和【原始背景资料】，为用户的【当前问题】生成一个结构清晰、内容详实、措辞专业的最终答案。请直接回答，不要复述思考过程。\n\n### 最终思考方案:\n{aggregated_thought}\n\n### 原始背景资料:\n{retrieved_context}\n\n### 当前问题:\n{user_question}\n\n### 你的最终回答:"""
    try:
        llm_payload["messages"] = [{"role": "user", "content": synthesis_prompt}]
        synthesis_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload, timeout=60)
        synthesis_response.raise_for_status()
        final_answer = synthesis_response.json()['choices'][0]['message']['content']
        return final_answer, reranked_chunks
    except Exception as e: return f"抱歉，在总结最终答案时遇到了问题: {e}", reranked_chunks


# --- 记忆检索与处理辅助函数 ---

def rerank_documents(query: str, documents: list, top_n: int = 5) -> list:
    if not documents or not cross_encoder:
        return documents[:top_n]
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores = cross_encoder.predict(pairs)
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:top_n]]
    except Exception as e:
        print(f"[警告] Re-ranker在预测时出错: {e}。将回退到简单排序。")
        return documents[:top_n]

def _get_common_context(user_question: str, user_state) -> Tuple[str, List[Document]]:
    from .state import UserState
    assert isinstance(user_state, UserState)

    if not user_state.long_term_db:
        return "记忆库尚未初始化。", []

    short_term_docs = user_state.short_term_cache.search(user_question, k=3)
    long_term_seed_docs = user_state.long_term_db.similarity_search_with_relevance_scores(user_question, k=7)

    retrieved_docs_map = {}
    for doc, score in long_term_seed_docs:
        if score > 0.6 or doc.metadata.get('type') == 'user_profile_fact':
            doc_id = doc.metadata.get('id')
            if doc_id not in retrieved_docs_map:
                mem_type = doc.metadata.get('type', 'general_note')
                content_prefix = ""
                if mem_type == 'user_profile_fact':
                    content_prefix = "【用户个人信息】: "
                elif mem_type == 'daily_summary':
                    content_prefix = "【每日工作摘要】: "
                elif mem_type == 'overall_personality':
                    content_prefix = "【用户全局画像】: "
                
                doc.page_content = content_prefix + doc.page_content
                retrieved_docs_map[doc_id] = doc

                if doc.metadata.get('source') == 'agentic_memory_structured':
                    original_mem_data_from_state = next((m for m in user_state.structured_memories if m['id'] == doc_id), None)
                    if original_mem_data_from_state:
                        linked_ids = original_mem_data_from_state.get('links', [])
                        for linked_id in linked_ids:
                            if linked_id not in retrieved_docs_map:
                                linked_mem_data_from_state = next((m for m in user_state.structured_memories if m['id'] == linked_id), None)
                                if linked_mem_data_from_state:
                                    linked_mem_type = linked_mem_data_from_state.get('type', 'general_note')
                                    linked_content_to_use = linked_mem_data_from_state.get('contextual_description', linked_mem_data_from_state.get('content', ''))
                                    linked_prefix = "【关联记忆】: "
                                    retrieved_docs_map[linked_id] = Document(
                                        page_content=linked_prefix + linked_content_to_use,
                                        metadata={"source": "agentic_memory_structured", "id": linked_id, "type": linked_mem_type, "strength": linked_mem_data_from_state.get('strength', 1.0)}
                                    )

    all_retrieved_docs = short_term_docs + list(retrieved_docs_map.values())
    unique_docs_map = {doc.page_content: doc for doc in all_retrieved_docs}

    reranked_chunks = rerank_documents(user_question, list(unique_docs_map.values()), top_n=6)

    final_chunks = []
    personal_facts_in_final = set()
    for chunk in reranked_chunks:
        if chunk.metadata.get('type') == 'user_profile_fact':
            personal_facts_in_final.add(chunk.metadata.get('id'))
        final_chunks.append(chunk)

    for doc in all_retrieved_docs:
        if doc.metadata.get('type') == 'user_profile_fact' and doc.metadata.get('id') not in personal_facts_in_final:
            final_chunks.append(doc)
            personal_facts_in_final.add(doc.metadata.get('id'))

    retrieved_context = "\n---\n".join([chunk.page_content for chunk in final_chunks]) if final_chunks else "无相关历史资料。"
    return retrieved_context, final_chunks


def process_user_input_for_facts(user_name: str, user_query: str, user_state) -> bool:
    """从用户输入中识别个人事实，并调用UserState方法进行存储。"""
    from .state import UserState
    assert isinstance(user_state, UserState)
    print(f"[记忆处理]: 正在从用户输入中识别个人事实...")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    llm_payload_base = {"model": DEEPSEEK_MODEL_NAME, "temperature": 0.2, "response_format": {"type": "json_object"}}
    personal_fact_check_prompt = f"请判断以下内容是否包含用户的个人信息（例如生日、偏好、住址、家人信息、工作、特定经历等）。如果是，请提取出完整的、简洁的个人事实句。请严格按照JSON格式返回：{{\"is_personal_fact\": true/false, \"extracted_fact\": \"\"}}。如果不是，'is_personal_fact'为false，'extracted_fact'为空字符串。\n内容: \"{user_query}\""

    llm_payload_base["messages"] = [{"role": "user", "content": personal_fact_check_prompt}]

    try:
        personal_check_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload_base, timeout=20)
        personal_check_response.raise_for_status()
        personal_check_data = json.loads(personal_check_response.json()['choices'][0]['message']['content'])

        if personal_check_data.get("is_personal_fact"):
            extracted_fact = personal_check_data.get("extracted_fact", "").strip()
            if extracted_fact:
                if user_state.add_personal_fact(extracted_fact):
                    print(f"[记忆处理]: 识别并暂存个人事实: {extracted_fact}")
                    return True
                else:
                    print(f"[记忆处理]: 个人事实 '{extracted_fact}' 已存在，跳过。")
        return False
    except Exception as e:
        print(f"[记忆处理][警告] 个人事实识别失败: {e}")
        return False


def process_structured_memory(user_name: str, new_content: str, content_type: str, user_state) -> bool:
    """处理并构建结构化记忆，并调用UserState方法进行存储。"""
    from .state import UserState
    assert isinstance(user_state, UserState)
    print(f"[A-Mem流水线]: 开始处理新的'{content_type}'结构化记忆...")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    llm_payload_base = {"model": DEEPSEEK_MODEL_NAME, "temperature": 0.2, "response_format": {"type": "json_object"}}

    note_construction_prompt = f'你是一个知识管理专家。请为以下信息生成一个结构化的记忆笔记。原始信息: "{new_content}"\n请严格按照JSON格式输出，包含keywords(3-5个核心名词或概念), tags(2-4个分类标签), 和 contextual_description(一句话精准概括):\n{{"keywords": [], "tags": [], "contextual_description": ""}}'
    llm_payload_base["messages"] = [{"role": "user", "content": note_construction_prompt}]

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload_base, timeout=45)
        response.raise_for_status()
        note_data = json.loads(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"[A-Mem][错误] 笔记构建失败: {e}")
        return False

    new_structured_memory = {
        "id": f"structured_mem_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "type": content_type,
        "content": new_content,
        "timestamp": datetime.datetime.now().isoformat(),
        "keywords": note_data.get("keywords", []),
        "tags": note_data.get("tags", []),
        "contextual_description": note_data.get("contextual_description", new_content),
        "links": [],
        "strength": 1.0
    }
    
    if content_type == "daily_summary":
        new_structured_memory["strength"] = 1.2
        new_structured_memory["id"] = f"daily_summary_{datetime.date.today().strftime('%Y-%m-%d')}"
    elif content_type == "overall_personality":
        new_structured_memory["id"] = f"overall_personality_{user_name}"
        new_structured_memory["strength"] = 2.0

    if user_state.long_term_db and user_state.structured_memories:
        neighbor_docs = user_state.long_term_db.similarity_search(new_structured_memory["contextual_description"], k=5)
        neighbor_notes_info = ""
        existing_structured_mem_dict = {m['id']:m for m in user_state.structured_memories if m['id'] != new_structured_memory['id']}
        for i, doc in enumerate(neighbor_docs):
            neighbor_id = doc.metadata.get('id')
            if doc.metadata.get('source') == 'agentic_memory_structured' and neighbor_id in existing_structured_mem_dict:
                desc_to_show = existing_structured_mem_dict[neighbor_id].get('contextual_description', existing_structured_mem_dict[neighbor_id].get('content', ''))
                neighbor_notes_info += f"{i+1}. (ID: {neighbor_id}) 描述: {desc_to_show}\n"

        if neighbor_notes_info:
            link_generation_prompt = f"你是一个知识网络构建师...请以JSON格式返回你认为应该链接的历史记忆ID列表 (例如 {{\"links_to_create\": [\"mem_id_1\"]}}):"
            llm_payload_base["messages"] = [{"role": "user", "content": link_generation_prompt}]
            try:
                link_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=llm_payload_base, timeout=45)
                link_response.raise_for_status()
                link_data = json.loads(link_response.json()['choices'][0]['message']['content'])
                new_structured_memory["links"] = link_data.get("links_to_create", [])
                print(f"[A-Mem流水线]: 链接生成完成, 新建链接: {new_structured_memory['links']}")
            except Exception as e: print(f"[A-Mem][错误] 链接生成失败: {e}")

    user_state.add_or_update_structured_memory(new_structured_memory)
    print(f"[A-Mem流水线]: 结构化记忆已暂存至内存。")
    return True


def reinforce_memory(user_name: str, used_chunks: List[Document], user_state) -> None:
    """调用UserState方法在内存中加固记忆。"""
    from .state import UserState
    assert isinstance(user_state, UserState)

    if not used_chunks: return
    print(f"[记忆加固]: 正在为 {len(used_chunks)} 条被回忆的记忆增强强度...")
    
    user_state.reinforce_memory_in_state(used_chunks)
    print("[记忆加固]: 记忆强度已在内存中更新。")


def reflect_and_update_memory(user_name: str, user_state) -> bool:
    """每日反思，生成摘要和画像，并通过process_structured_memory暂存到内存。"""
    from .state import UserState
    assert isinstance(user_state, UserState)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    summary_exists = any(mem.get('type') == 'daily_summary' and mem.get('id') == f"daily_summary_{today_str}" for mem in user_state.structured_memories)
    
    current_day_history_turns = [
        t for t in user_state.conversation_history 
        if 'timestamp' in t and datetime.datetime.fromisoformat(t['timestamp']).date() == datetime.date.today()
    ]

    if not current_day_history_turns:
        print("[反思机制]: 今日无对话，跳过反思。")
        return False

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [], "temperature": 0.0}
    made_update = False

    if not summary_exists:
        print("[反思机制]: 检测到今日新对话，开始进行每日摘要生成...")
        conversation_text = "\n".join([f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in current_day_history_turns])
        summary_prompt = f"""你是一个专业的会议纪要员...输出格式:\n```json\n{{\n    \"待办事项\": [],\n    \"关键决策\": [],\n    \"核心问题与发现\": []\n}}\n```"""
        payload['messages'] = [{"role": "user", "content": summary_prompt}]
        payload['response_format'] = {"type": "json_object"}
        
        try:
            summary_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
            summary_response.raise_for_status()
            daily_summary_obj = json.loads(summary_response.json()['choices'][0]['message']['content'])
            daily_summary_content = (f"待办事项: {'; '.join(daily_summary_obj.get('待办事项', ['无']))}\n"
                                     f"关键决策: {'; '.join(daily_summary_obj.get('关键决策', ['无']))}\n"
                                     f"核心问题与发现: {'; '.join(daily_summary_obj.get('核心问题与发现', ['无']))}")
            if process_structured_memory(user_name, daily_summary_content, "daily_summary", user_state):
                made_update = True
                print("[反思机制]: 每日摘要生成并存储到 A-Mem 成功。")
        except Exception as e:
            print(f"[错误] 生成每日摘要失败: {e}")

    all_summaries_content = [f"日期 {mem['id'].replace('daily_summary_', '')}:\n{mem.get('content', '')}" for mem in user_state.structured_memories if mem.get('type') == 'daily_summary']
    all_summaries_text = "\n\n".join(all_summaries_content)

    if all_summaries_text:
        print("[反思机制]: 准备更新全局用户画像...")
        portrait_prompt = f"""你是一位资深的人力资源(HR)和职业规划顾问...**专业画像总结**:"""
        payload['messages'] = [{"role": "user", "content": portrait_prompt}]
        payload.pop('response_format', None)
        
        try:
            profile_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
            profile_response.raise_for_status()
            professional_profile = profile_response.json()['choices'][0]['message']['content'].strip()
            if process_structured_memory(user_name, professional_profile, "overall_personality", user_state):
                made_update = True
                print("[反思机制]: 全局用户画像更新并存储到 A-Mem 成功。")
        except Exception as e:
            print(f"[错误] 更新全局画像失败: {e}")

    return made_update