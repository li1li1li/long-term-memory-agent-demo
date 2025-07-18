import os
import json
import requests
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

## ======================================================================================
# 1. 配置区域
'''
# ======================================================================================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# --- 在这里替换成你的真实 API Key ---
DEEPSEEK_API_KEY = "sk-a7ad78960dcf4f338385e20cd59534cb" 
DEEPSEEK_MODEL_NAME = "deepseek-chat"

'''
DEEPSEEK_API_URL = "http://localhost:11434/v1/chat/completions"
DEEPSEEK_API_KEY = "ollama"  # API Key不再需要，可以填写任意字符作为占位符
DEEPSEEK_MODEL_NAME = "qwen2:7b" # 必须与你在Ollama中运行的模型名一致

# 本地向量化模型的配置 (这个模型会自动从Hugging Face下载并运行在你的GPU上)
EMBEDDING_MODEL_NAME = "./models/all-MiniLM-L6-v2"

# 持久化存储路径配置
PROJECT_DIR = "./memoryBank-demo/"  #  可以根据你的需要修改路径，'.'表示当前目录
KNOWLEDGE_BASE_FILE = os.path.join(PROJECT_DIR, "my_knowledge.txt")
MEMORY_BANK_FILE = os.path.join(PROJECT_DIR, "memory_bank.txt")
FAISS_INDEX_DIR = os.path.join(PROJECT_DIR, "my_faiss_index")

# 对话历史窗口大小配置 (表示保留最近的N轮对话作为短期上下文)
HISTORY_WINDOW_SIZE = 2 


## ======================================================================================
# 2. 核心功能函数
# ======================================================================================

def build_knowledge_base(force_rebuild=False):
    """
    【离线功能】读取知识库和记忆库文件，将其处理成向量并构建FAISS索引。
    """
    if force_rebuild and os.path.exists(FAISS_INDEX_DIR):
        print("[后台任务：检测到强制重建指令，正在删除旧索引...]")
        shutil.rmtree(FAISS_INDEX_DIR)

    if os.path.exists(FAISS_INDEX_DIR):
        print(f"[初始化：知识库索引 '{FAISS_INDEX_DIR}' 已存在，跳过构建过程。]")
        return

    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    # 确保基础知识库文件存在
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"[初始化：未找到知识库文件，正在创建示例文件: {KNOWLEDGE_BASE_FILE}]")
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            f.write("RAG是检索增强生成的缩写，它先从知识库检索信息，再交给大模型生成答案。它的优点是可以利用外部知识，减少模型幻觉。\n")
            f.write("FAISS是一个用于高效向量搜索的库，由Facebook AI开发。\n")

    # 确保记忆库文件存在
    if not os.path.exists(MEMORY_BANK_FILE):
        print(f"[初始化：未找到记忆库文件，正在创建空文件: {MEMORY_BANK_FILE}]")
        with open(MEMORY_BANK_FILE, "w", encoding="utf-8") as f:
            f.write("这是一个记忆库，用于存储关于用户的核心事实。\n")

    # 同时读取知识库和记忆库
    print("[初始化：正在读取知识库和记忆库文件...]")
    knowledge_text = ""
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        knowledge_text += f.read()
    with open(MEMORY_BANK_FILE, "r", encoding="utf-8") as f:
        knowledge_text += "\n" + f.read()

    # 文本分割
    print("[初始化：正在进行文本分割...]")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    text_chunks = text_splitter.split_text(knowledge_text)
    print(f"[初始化：文本被分割成 {len(text_chunks)} 个小块。]")

    # 文本向量化
    print("[初始化：正在进行文本向量化 (首次运行会下载模型，请稍候)...]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}  # 明确指定使用GPU
    )

    # 构建并保存 FAISS 索引
    print("[初始化：正在构建并保存FAISS索引...]")
    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    print(f"[初始化：知识库索引构建完成！已保存至 '{FAISS_INDEX_DIR}']")


def load_vector_db(embeddings_model):
    """一个辅助函数，用于加载FAISS索引"""
    print("[系统：正在加载本地知识库索引...]")
    if not os.path.exists(FAISS_INDEX_DIR):
        print("[错误：FAISS索引不存在，请先运行程序生成。]")
        return None
    try:
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings_model, allow_dangerous_deserialization=True)
        print("[系统：知识库加载完成！]")
        return db
    except Exception as e:
        print(f"[错误：加载索引时出错] {e}")
        return None


def answer_question(user_question, db, history):
    """
    【在线功能】融合了RAG、长期记忆和短期上下文的问答函数
    """
    print("[主任务：正在融合知识、记忆和上下文进行检索...]")
    retrieved_chunks = db.similarity_search(user_question, k=3)

    if not retrieved_chunks:
        context = "无相关信息"
    else:
        context = "\n---\n".join([chunk.page_content for chunk in retrieved_chunks])

    history_prompt = ""
    for turn in history:
        if turn['role'] == 'user':
            history_prompt += f"历史提问: {turn['content']}\n"
        elif turn['role'] == 'assistant':
            history_prompt += f"历史回答: {turn['content']}\n\n"

    prompt = f"""你是一个专业的问答机器人。请综合分析【参考资料】和【历史对话】来回答【当前问题】。
你的回答需要自然、连贯，就像一个真正的聊天伙伴。
如果【参考资料】和【历史对话】都无法提供足够信息，就回答："抱歉，我无法回答这个问题。"

---
【参考资料】:
{context}
---
【历史对话】:
{history_prompt}
---

【当前问题】:
{user_question}

【你的回答】:
"""

    messages_payload = history + [{"role": "user", "content": prompt}]
    
    print("[主任务：正在请求 DeepSeek API 回答问题...]")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": DEEPSEEK_MODEL_NAME,
        "messages": messages_payload,
        "temperature": 0.3
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    api_result = response.json()
        
        # --- 核心修改：使用 .get() 方法安全地解析JSON ---
    choices = api_result.get('choices', [])
    if choices:
        message = choices[0].get('message', {})
        content = message.get('content', '') # 如果content不存在，返回空字符串
        return content
    else:
        return "抱歉，AI模型返回了空的回复。"



'''
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        api_result = response.json()
        
        # --- 核心修改：使用 .get() 方法安全地解析JSON ---
        choices = api_result.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            content = message.get('content', '') # 如果content不存在，返回空字符串
            return content
        else:
            return "抱歉，AI模型返回了空的回复。"
        # ------------------------------------------------

    except Exception as e:
        print(f"[错误：调用主任务API时发生错误] {e}")
        return "抱歉，请求AI服务时遇到了问题。"
'''

def extract_and_save_memory(conversation_turn, existing_memories):
    """
    接收一轮对话，调用API提取事实，并存入长期记忆。
    """
    print("\n[后台任务：正在进行记忆提取反思...]")
    
    # --- 核心修改：对要嵌入的字符串进行转义，防止破坏JSON结构 ---
    # 将 user_input 和 assistant_output 中的双引号和反斜杠进行转义
    user_input_escaped = conversation_turn['user'].replace('\\', '\\\\').replace('"', '\\"')
    assistant_output_escaped = conversation_turn['assistant'].replace('\\', '\\\\').replace('"', '\\"')
    # -----------------------------------------------------------

    # 1. 构建提取事实的Prompt，使用转义后的安全字符串
    extraction_prompt = f"""你是一个信息提取助手。请从下面这轮【对话】中，提取出关于用户的、值得长期记住的**核心事实**。
请遵循以下规则：
- 只提取关于用户本人的事实（例如：用户的名字、家乡、职业、爱好等）。
- 如果是已经知道或者不重要的信息，忽略它。
- 以简洁的陈述句形式输出，每条事实占一行。
- 如果对话中没有值得记录的新事实，请只回答"无"，不要有任何其他多余的内容。

【对话】:
用户: "{user_input_escaped}"
机器人: "{assistant_output_escaped}"

【提取的核心事实】:
"""

    # 2. 调用API进行事实提取
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {
        "model": DEEPSEEK_MODEL_NAME,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "temperature": 0.0
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        api_result = response.json()
        
        extracted_facts_text = ""
        choices = api_result.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            extracted_facts_text = message.get('content', '').strip()
        
        if extracted_facts_text and extracted_facts_text != "无":
            new_facts = [fact.strip() for fact in extracted_facts_text.split('\n') if fact.strip()]
            
            memory_updated = False
            with open(MEMORY_BANK_FILE, "a+", encoding="utf-8") as f:
                for fact in new_facts:
                    if fact not in existing_memories:
                        print(f"[后台任务：发现新记忆 -> '{fact}']")
                        f.write(fact + "\n")
                        existing_memories.add(fact)
                        memory_updated = True
            
            return memory_updated
        else:
            print("[后台任务：未发现值得记忆的新信息。]")
            return False
    except Exception as e:
        print(f"[后台任务：记忆提取时发生错误] {e}")
        return False
## ======================================================================================
# 3. 主程序入口
# ======================================================================================

if __name__ == "__main__":
    # 首次运行或需要时，构建知识库
    build_knowledge_base()

    # 在程序开始时，一次性加载所有已知的记忆，用于去重检查
    with open(MEMORY_BANK_FILE, "r", encoding="utf-8") as f:
        known_memories = set(line.strip() for line in f if line.strip())

    print("\n[系统：正在加载向量化模型...]")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cuda'})
    db = load_vector_db(embeddings)
    if db is None:
        exit()
    
    conversation_history = []
    print("\n========================================")
    print("欢迎使用带隐式记忆的AI聊天伙伴！")
    print("========================================")

    while True:
        question = input("\n你: ")
        if question.strip().lower() in ["退出", "exit", "quit"]:
            print("机器人: 感谢使用，再见！")
            break
        if not question.strip():
            continue

        # 调用问答函数获取答案
        final_answer = answer_question(question, db, conversation_history)
        print(f"\n机器人: {final_answer}")

        # 在回答后，自动进行记忆提取和保存
        current_turn = {"user": question, "assistant": final_answer}
        if extract_and_save_memory(current_turn, known_memories):
            print("[后台任务：长期记忆已更新，正在重建知识索引...]")
            build_knowledge_base(force_rebuild=True)
            db = load_vector_db(embeddings)
            print("[后台任务：知识索引重建完成！]")

        # 更新短期对话历史
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": final_answer})
        if len(conversation_history) > HISTORY_WINDOW_SIZE * 2:
            conversation_history = conversation_history[-HISTORY_WINDOW_SIZE * 2:]