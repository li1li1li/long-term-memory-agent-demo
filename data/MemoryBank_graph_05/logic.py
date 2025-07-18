# logic.py
import datetime
import json
from typing import Tuple, List
from langchain.docstore.document import Document
import re
import config
from state import AppState
from llm_wrappers import LLMManager

llm = LLMManager()

def generate_conversational_response(user_question: str, user_state: AppState, deep_mode: bool) -> Tuple[str, List[Document]]:
    """根据模式选择，生成对话回复。"""
    if deep_mode:
        print("[AI助手模式]: 启动深度思考模式。")
        return _execute_deep_thought_mode(user_question, user_state)
    else:
        print("[AI助手模式]: 启动标准对话模式。")
        return _execute_standard_mode(user_question, user_state)

def _execute_standard_mode(user_question: str, user_state: AppState) -> Tuple[str, List[Document]]:
    """标准模式：基于统一检索的上下文直接回答。"""
    # 1. 获取统一上下文
    retrieved_context, used_docs = user_state.memory_manager.get_combined_context(user_question)
    
    # 2. 准备 Prompt
    history_prompt = user_state.get_full_history_text()
    
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
    # 3. 调用LLM
    response = llm.generate_response([{"role": "user", "content": persona_prompt}], temperature=0.7)
    return response['content'], used_docs
def _find_and_parse_json(text: str):
    """从可能包含额外文本的字符串中查找并解析第一个有效的JSON对象。"""
    # 查找被```json ... ```包裹的代码块
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果失败，尝试找到第一个 '{' 和最后一个 '}' 之间的内容
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None # 彻底失败
        return None


def _execute_deep_thought_mode(user_question: str, user_state: AppState) -> Tuple[str, List[Document]]:
    """
    深度思考模式：执行GoT流程，增加了数据净化和强化版Prompt。
    """
    print("[逻辑]: 启动深度思考模式...")
    retrieved_context, used_docs = user_state.memory_manager.get_combined_context(user_question)
    history_prompt = user_state.get_full_history_text()

    # --- 数据净化步骤 ---
    # 将动态获取的上下文和历史记录进行封装，防止其内容干扰主指令
    safe_context = f"<背景资料>\n{retrieved_context}\n</背景资料>"
    safe_history = f"<对话历史>\n{history_prompt}\n</对话历史>"


    # --- 1. 思想生成阶段 (使用净化数据和强化版Prompt) ---
    generation_prompt = f"""
你是一位顶级的、富有逻辑性的思想家。你的任务是为用户的【当前问题】生成解决问题的思考路径。

**重要指令**:
1.  你必须严格、精确地按照我要求的格式输出。
2.  输出内容只能包含思考路径，绝对不能包含任何额外的解释、对话、前言或总结。
3.  你必须分析我提供给你的被`<标签>`包裹的背景资料，但你自己的输出中不能使用`<标签>`。

**这是一个输出格式的优秀范例**:
[思考路径 1]
分析问题的核心要素。首先，拆解问题为A、B、C三个部分。然后，分别评估每个部分的重要性。
[思考路径 2]
从相反的角度思考。如果问题不成立，需要满足哪些条件？这些条件现实吗？这可以帮助我们发现思维盲点。
[思考路径 3]
寻找外部案例与类比。历史上或行业内是否有类似的问题？他们是如何解决的？我们可以从中借鉴什么？

---
{safe_context}

{safe_history}
---
### 当前问题:
{user_question}

---
请严格按照上面的范例格式，为当前问题生成你的思考路径：
"""
    
    # 打印最终发送给LLM的完整Prompt，用于调试
    print("\n\n" + "#"*20 + " 发送给LLM的最终Prompt " + "#"*20)
    print(generation_prompt)
    print("#"*61 + "\n\n")

    thoughts = {}
    max_retries = 2
    for attempt in range(max_retries):
        print(f"[深度思考] 正在生成思考路径... (尝试 {attempt + 1}/{max_retries})")
        generation_response = llm.generate_response([{"role": "user", "content": generation_prompt}], temperature=0.7)
        generated_thoughts_text = generation_response['content']
        
        # 打印LLM返回的原始内容，用于调试
        print("\n" + "="*20 + " LLM原始返回内容 " + "="*20)
        print(generated_thoughts_text)
        print("="*58 + "\n")

        temp_thoughts = {}
        for i in range(1, 4):
            start_tag = f"[思考路径 {i}]"
            end_tag = f"[思考路径 {i+1}]" if i < 3 else None
            
            start_index = generated_thoughts_text.find(start_tag)
            if start_index != -1:
                start_index += len(start_tag)
                end_index = generated_thoughts_text.find(end_tag) if end_tag else len(generated_thoughts_text)
                temp_thoughts[f'path_{i}'] = generated_thoughts_text[start_index:end_index].strip()
        
        if temp_thoughts:
            thoughts = temp_thoughts
            print("[深度思考] 成功解析思考路径。")
            break
        else:
            print(f"[深度思考][警告] 第 {attempt + 1} 次尝试未能解析思考路径。")

    if not thoughts: 
        return f"抱歉，在进行头脑风暴时遇到了问题: 多次尝试后，模型仍未能按预期格式返回思考路径。", used_docs

    # --- 2. 思想聚合阶段 ---
    aggregation_input = "".join([f"### 备选思路 {i}:\n{path_text}\n\n" for i, path_text in enumerate(thoughts.values(), 1)])
    aggregation_prompt = f"""你是一位顶级的策略分析师。请综合、提炼下面的多个【备选思路】，形成一个唯一的、逻辑更严密、内容更全面、结构更清晰的【最终思考方案】。\n\n{aggregation_input}\n---\n### 当前问题:\n{user_question}\n\n### 你的最终思考方案:"""
    aggregation_response = llm.generate_response([{"role": "user", "content": aggregation_prompt}], temperature=0.4)
    aggregated_thought = aggregation_response['content']

    # --- 3. 最终答案生成阶段 ---
    synthesis_prompt = f"""你是一位高级AI知识助手。请严格依据下面提供的【最终思考方案】和【原始背景资料】，为用户的【当前问题】生成一个结构清晰、内容详实、措辞专业的最终答案。\n\n### 最终思考方案:\n{aggregated_thought}\n\n### 原始背景资料:\n{retrieved_context}\n\n### 当前问题:\n{user_question}\n\n### 你的最终回答:"""
    synthesis_response = llm.generate_response([{"role": "user", "content": synthesis_prompt}], temperature=0.5)
    
    return synthesis_response['content'], used_docs

def process_user_input_for_facts(user_name: str, user_query: str, user_state: AppState) -> bool:
    """从用户输入中识别并存储个人事实。"""
    print(f"[逻辑]: 正在从 '{user_query}' 中识别个人事实...")
    prompt = f"请判断以下内容是否包含用户的个人信息（例如生日、偏好、住址、家人信息、工作、特定经历等）。如果是，请提取出完整的、简洁的个人事实句。请严格按照JSON格式返回：{{\"is_personal_fact\": true/false, \"extracted_fact\": \"\"}}。\n内容: \"{user_query}\""
    response = llm.generate_response([{"role": "user", "content": prompt}], use_json_format=True)
    data = response['content']
    if isinstance(data, dict) and data.get("is_personal_fact") and data.get("extracted_fact"):
        fact_content = data["extracted_fact"]
        fact_doc = Document(page_content=fact_content, metadata={"source": "faiss", "type": "user_profile_fact"})
        user_state.memory_manager.add_fact_memory(fact_doc)
        print(f"[逻辑]: 识别并存储了个人事实: {fact_content}")
        return True
    return False

def reflect_and_update_memory(user_name: str, user_state: AppState) -> bool:
    """每日反思机制：总结当日对话，更新用户全局画像。"""
    print("[逻辑]: 正在执行每日反思...")
    today_history = user_state.get_today_history()
    if not today_history:
        print("[逻辑]: 今日无对话，跳过反思。")
        return False
        
    conversation_text = "\n".join([f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in today_history])
    
    # 1. 生成每日摘要
    summary_prompt = f"""你是一个专业的会议纪要员和项目助理。请严格按照以下要求，从对话内容中提取结构化的工作信息。
    **重要指令**:
1.  "待办事项"、"关键决策"、"核心问题与发现"字段的值必须是字符串列表
2.  每个列表项应该是简洁的纯文本描述，不要包含JSON对象

**指令**:
1.  **重点关注**与工作、学习、项目、决策相关的实质性内容。
2.  *
3.  如果某个类别下没有内容，请明确写出“无”。

**对话内容**:
{conversation_text}

**输出格式**:
```json
{{
    "待办事项": [
        "完成季度报告",
        "安排团队培训"
    ],
    "关键决策": [
        "采用新技术栈",
        "调整项目时间表"
    ],
    "核心问题与发现": [
        "服务器性能瓶颈",
        "用户需求变化"
    ]
}}
```"""
    summary_response = llm.generate_response([{"role": "user", "content": summary_prompt}], use_json_format=True)
    summary_obj = summary_response['content']
    if isinstance(summary_obj, dict):
        summary_content = (f"待办事项: {'; '.join(summary_obj.get('待办事项', ['无']))}\n"
                         f"关键决策: {'; '.join(summary_obj.get('关键决策', ['无']))}\n"
                         f"核心问题与发现: {'; '.join(summary_obj.get('核心问题与发现', ['无']))}")
        summary_doc = Document(page_content=summary_content, metadata={"source": "faiss", "type": "daily_summary", "id": f"daily_summary_{datetime.date.today().strftime('%Y-%m-%d')}"})
        user_state.memory_manager.add_fact_memory(summary_doc)
        print("[逻辑]: 每日摘要已生成并存储。")

    # 2. 更新全局用户画像
    all_summaries = user_state.memory_manager.get_all_facts_of_type("daily_summary")
    if all_summaries:
        all_summaries_text = "\n\n".join([f"日期 {doc.metadata.get('id', '').replace('daily_summary_', '')}:\n{doc.page_content}" for doc in all_summaries])
        
        portrait_prompt = f"""你是一位资深的人力资源(HR)和职业规划顾问。请基于以下长期的工作摘要，为用户【{user_name}】提炼出一个专业的、以能力和目标为导向的画像。

**指令**:
1.  分析用户的核心能力、展现出的技能和知识领域。
2.  推断用户可能的长期职业目标或当前项目的核心目的。
3.  识别用户工作模式中潜在的风险或可以改进的地方。
4.  用简洁、专业的语言总结，分点阐述。

**长期工作摘要历史**:
{all_summaries_text}

**专业画像总结**:"""
        profile_response = llm.generate_response([{"role": "user", "content": portrait_prompt}])
        profile_content = profile_response['content']
        profile_doc = Document(page_content=profile_content, metadata={"source": "faiss", "type": "overall_personality", "id": f"overall_personality_{user_name}"})
        user_state.memory_manager.add_fact_memory(profile_doc)
        print("[逻辑]: 全局用户画像已更新。")
        return True
        
    return False