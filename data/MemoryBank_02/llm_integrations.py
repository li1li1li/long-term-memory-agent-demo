# llm_integrations.py

import json
import requests
import datetime
from typing import List, Set, Tuple

# LangChain 和本地模块的导入
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
import config

def rerank_documents(query: str, documents: List[Document], top_n: int = 2) -> List[Document]:
    """使用CrossEncoder对检索到的文档进行重排序。"""
    if not documents: return []
    print(f"[LLM交互]: 正在使用CrossEncoder对 {len(documents)} 个文档进行重排序...")
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        cross_encoder = CrossEncoder(config.RERANKER_MODEL_PATH)
        scores = cross_encoder.predict(pairs)
        doc_scores = sorted(list(zip(documents, scores)), key=lambda x: x[1], reverse=True)
        print(f"[LLM交互]: 重排序完成，选取Top {top_n} 的结果。")
        return [doc for doc, score in doc_scores[:top_n]]
    except Exception as e:
        print(f"[警告] Re-ranker执行失败: {e}。将返回原始文档。")
        return documents[:top_n]

def answer_question(user_question: str, db: FAISS, history: List[dict], user_name: str) -> Tuple[str, List[Document]]:
    """根据上下文和记忆生成答案。"""
    retrieved_chunks = db.similarity_search(user_question, k=5)
    reranked_chunks = rerank_documents(user_question, retrieved_chunks, top_n=2)
    unique_contents: Set[str] = {chunk.page_content for chunk in reranked_chunks}
    retrieved_context = "\n---\n".join(unique_contents) if unique_contents else "无相关历史资料。"
    with open(config.MEMORY_FILE, 'r', encoding='utf-8') as f:
        professional_profile = json.load(f).get(user_name, {}).get('overall_personality', '用户工作偏好尚不明确。')
    history_prompt = "".join([f"{'用户' if t['role']=='user' else 'AI助手'}: {t['content']}\n" for t in history])
    prompt = f"""你是一个高级AI知识助手，专为学术研究和办公场景设计。
### 指令
1.  **逻辑严谨**: 你的回答必须基于提供的资料，如果资料不足，请明确指出。
2.  **专业客观**: 回答应专业、中立，避免使用朋友般的口吻。
3.  **结构清晰**: 对复杂问题，优先使用列表、点号呈现答案。
4.  **简洁精炼**: 综合所有信息后，请提供核心要点，避免直接复述原文。
---
### 背景资料：长期记忆库
{retrieved_context}
---
### 背景资料：关于用户的工作偏好
{professional_profile}
---
### 背景资料：近期对话上下文
{history_prompt}
---
【用户当前问题】:
{user_question}
【你的回答】:"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.API_KEY}"}
    payload = {"model": config.MODEL_NAME, "messages": history + [{"role": "user", "content": prompt}], "temperature": 0.4}
    try:
        response = requests.post(config.API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        final_answer = response.json()['choices'][0]['message']['content']
        return final_answer, reranked_chunks
    except Exception as e:
        return f"抱歉，连接AI服务时遇到了问题: {e}", []

def reflect_and_update_memory(user_name: str) -> bool:
    """调用LLM进行反思，生成每日摘要和更新用户画像。"""
    with open(config.MEMORY_FILE, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)
    user_history = memory_data.get(user_name, {}).get('history', {})
    if not user_history: return False
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if not user_history.get(today_str): return False
    conversation_text = "\n".join([f"用户: {t['query']}\nAI助手: {t['response']}" for t in user_history[today_str]])
    summary_prompt = f"请以客观、精炼的语言，从以下对话中提取关键信息、待办事项(To-do)、或重要决策点。如果无特殊要点，则总结对话的核心议题。\n\n对话内容:\n{conversation_text}\n\n今日工作要点总结:"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.API_KEY}"}
    payload = {"model": config.MODEL_NAME, "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.1}
    made_update = False
    try:
        daily_summary = requests.post(config.API_URL, headers=headers, json=payload).json()['choices'][0]['message']['content'].strip()
        if 'summary' not in memory_data[user_name]: memory_data[user_name]['summary'] = {}
        memory_data[user_name]['summary'][today_str] = {"content": daily_summary}
        made_update = True
    except Exception as e: print(f"[LLM交互][错误] 生成每日摘要失败: {e}")
    all_summaries = "\n".join([f"日期 {d}: {s.get('content')}" for d, s in memory_data[user_name].get('summary', {}).items()])
    portrait_prompt = f"请根据以下长期的工作要点摘要，分析并总结用户【{user_name}】的专业领域、研究方向、工作习惯和核心关注点。输出一个简短、客观的第三人称工作画像描述。\n\n工作要点历史:\n{all_summaries}\n\n更新后的用户工作画像:"
    payload['messages'][0]['content'] = portrait_prompt
    try:
        professional_profile = requests.post(config.API_URL, headers=headers, json=payload).json()['choices'][0]['message']['content'].strip()
        memory_data[user_name]['overall_personality'] = professional_profile
        made_update = True
    except Exception as e: print(f"[LLM交互][错误] 更新全局画像失败: {e}")
    if made_update:
        with open(config.MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
    return made_update