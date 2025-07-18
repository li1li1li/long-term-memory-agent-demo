import os
import json
import requests
import shutil
import datetime
from typing import List

# 核心依赖：我们只依赖这些在您环境中已经存在的、不太可能冲突的库
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# ======================================================================================
# 0. 自定义的、更稳定的嵌入类 (核心修改，用于取代有问题的HuggingFaceEmbeddings)
# ======================================================================================
class StableHuggingFaceEmbeddings:
    """
    一个更稳定的自定义嵌入类，它直接、简单地使用 sentence-transformers 库的核心功能，
    以此来完全绕过 langchain 中那段与您服务器环境不兼容的、复杂的内部导入代码。
    """
    def __init__(self, model_name_or_path: str, device: str = 'cuda'):
        """
        在初始化时，直接加载SentenceTransformer模型。
        参数:
            model_name_or_path (str): 您本地模型的路径，例如 './models/all-MiniLM-L6-v2'
            device (str): 模型运行的设备，例如 'cuda' 或 'cpu'
        """
        try:
            print(f"[自定义嵌入类]: 正在从 '{model_name_or_path}' 加载嵌入模型...")
            # 直接创建SentenceTransformer实例，这是最核心的调用
            self.model = SentenceTransformer(model_name_or_path, device=device)
            print("[自定义嵌入类]: 嵌入模型加载成功。")
        except Exception as e:
            print(f"[自定义嵌入类]: 致命错误：加载SentenceTransformer模型失败。错误信息: {e}")
            print("这通常意味着 sentence-transformers 或其依赖的 torch/transformers 库在您的环境中存在无法解决的冲突。")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表创建嵌入向量，供FAISS索引使用。"""
        print(f"[自定义嵌入类]: 正在为 {len(texts)} 个文档片段创建嵌入...")
        # encode方法返回numpy数组，需要转换为list
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        print("[自定义嵌入类]: 文档嵌入创建完成。")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """为单个查询文本创建嵌入向量，供FAISS搜索使用。"""
        # encode方法返回numpy数组，需要转换为list
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding

# ======================================================================================
# 1. 配置区域
# ======================================================================================
'''DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = "sk-a7ad78960dcf4f338385e20cd59534cb"  # <--- 在这里替换成你的真实 API Key
DEEPSEEK_MODEL_NAME = "deepseek-chat"
'''
DEEPSEEK_API_URL = "http://localhost:11434/v1/chat/completions"
DEEPSEEK_API_KEY = "ollama"  # API Key不再需要，可以填写任意字符作为占位符
DEEPSEEK_MODEL_NAME = "qwen2:7b" # 必须与你在Ollama中运行的模型名一致

EMBEDDING_MODEL_NAME = "./models/all-MiniLM-L6-v2"  # 您本地已经下载好的模型路径

PROJECT_DIR = "./memoryBank-demo/"
MEMORY_FILE = os.path.join(PROJECT_DIR, "memory.json")
FAISS_INDEX_BASE_DIR = os.path.join(PROJECT_DIR, "faiss_indices")

HISTORY_WINDOW_SIZE = 5

# ======================================================================================
# 2. 核心功能函数 (现在将使用我们新的 StableHuggingFaceEmbeddings)
# ======================================================================================

def initialize_memory_file(memory_path, user_name):
    """
    【优化后的版本】
    初始化内存文件。如果文件不存在，则创建新文件并添加用户。
    如果文件已存在，则检查当前用户是否存在，如果不存在，则向文件中追加新用户信息。
    """
    # 确保项目目录存在
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    # 为新用户准备的默认数据结构
    new_user_data = {
        "name": user_name,
        "summary": {},
        "personality": {},
        "overall_history": "暂无全局历史摘要。",
        "overall_personality": "用户性格尚不明确，需要通过更多对话来了解。",
        "history": {}
    }

    if not os.path.exists(memory_path):
        # --- 场景1: 内存文件不存在 ---
        # 直接创建新文件，并写入第一个用户的数据
        print(f"[初始化]: 未找到内存文件，正在为新用户 '{user_name}' 创建: {memory_path}")
        initial_data = {user_name: new_user_data}
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=4)
    else:
        # --- 场景2: 内存文件已存在 ---
        # 打开文件，检查当前用户是否已经是其中的一员
        with open(memory_path, "r+", encoding="utf-8") as f:
            try:
                memory_data = json.load(f)
            except json.JSONDecodeError:
                # 如果文件是空的或损坏的，当作新文件处理
                memory_data = {}

            if user_name not in memory_data:
                # 如果当前用户是新用户，则将他的信息添加到已加载的数据中
                print(f"[初始化]: 内存文件已存在，正在为新用户 '{user_name}' 添加记录...")
                memory_data[user_name] = new_user_data
                
                # 将更新后的全部内容写回文件
                f.seek(0)  # 指针移到文件开头
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
                f.truncate() # 清除可能存在的旧文件尾部多余内容
            else:
                # 如果用户已存在，则什么都不做
                print(f"[初始化]: 在内存文件中找到现有用户 '{user_name}' 的记录。")

def load_and_index_memory(memory_file_path, user_name, embeddings_model, force_rebuild=False):
    faiss_index_path = os.path.join(FAISS_INDEX_BASE_DIR, user_name)
    if force_rebuild and os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
        
    if os.path.exists(faiss_index_path) and not force_rebuild:
        print(f"[系统]: 正在从 '{faiss_index_path}' 为用户 '{user_name}' 加载现有FAISS索引。")
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        
        # 【关键修改点 1/2】: 加载db对象后，手动将其内部的查询函数指向我们类中正确的方法
        db.embedding_function = embeddings_model.embed_query
        print("[系统]: FAISS 对象的查询函数已修正。")
        return db

    print(f"[系统]: 正在为用户 '{user_name}' 构建新的FAISS索引...")
    with open(memory_file_path, 'r', encoding='utf-8') as f:
        user_memory = json.load(f).get(user_name, {})
    if not user_memory:
        print(f"[警告]: 在内存文件中未找到用户 '{user_name}' 的数据。")
        return None

    docs_to_index = []
    # 索引所有类型的记忆
    for date, daily_history in user_memory.get('history', {}).items():
        for turn in daily_history:
            docs_to_index.append(Document(page_content=f"日期 {date} 的对话: 用户: '{turn['query']}', AI: '{turn['response']}'", metadata={"source": "history"}))
    for date, summary in user_memory.get('summary', {}).items():
        docs_to_index.append(Document(page_content=f"日期 {date} 的摘要: {summary.get('content', '')}", metadata={"source": "summary"}))
    docs_to_index.append(Document(page_content=f"全局用户画像: {user_memory.get('overall_personality', '')}", metadata={"source": "overall_personality"}))

    if not docs_to_index:
        print("[警告]: 无内容可供索引，将创建一个空索引。")
        db = FAISS.from_documents([Document(page_content=" ")], embeddings_model)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs_to_index)
        db = FAISS.from_documents(split_docs, embeddings_model)
    
    # 【关键修改点 2/2】: 创建db对象后，手动将其内部的查询函数指向我们类中正确的方法
    db.embedding_function = embeddings_model.embed_query
    print("[系统]: FAISS 对象的查询函数已修正。")
    
    os.makedirs(faiss_index_path, exist_ok=True)
    db.save_local(faiss_index_path)
    print(f"[系统]: 索引构建完成！已保存至 '{faiss_index_path}'")
    return db
def answer_question(user_question, db, history, user_name):
    retrieved_chunks = db.similarity_search(user_question, k=4) # 【可选优化】: 稍微增加检索数量，为专业问题提供更多上下文
    retrieved_context = "\n---\n".join([chunk.page_content for chunk in retrieved_chunks]) if retrieved_chunks else "无相关历史资料。"
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        # 【修改】: 变量名从 user_portrait 改为 professional_profile，更符合新角色
        professional_profile = json.load(f).get(user_name, {}).get('overall_personality', '用户工作偏好尚不明确。')
    history_prompt = "".join([f"{'用户' if t['role']=='user' else 'AI助手'}: {t['content']}\n" for t in history])

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修改点：重新设计主Prompt ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    prompt = f"""你是一个高级AI知识助手，专为学术研究和办公场景设计。

你的核心任务是：基于提供的【背景资料】和【历史对话】，以清晰、严谨、结构化的方式回答用户的【当前问题】。

### 指令
1.  **逻辑严谨**: 你的回答必须基于提供的资料，如果资料不足，请明确指出。禁止提供猜测或未经证实的信息。
2.  **专业客观**: 回答应专业、中立，避免使用朋友般的口吻或过多的情感表达。
3.  **结构清晰**: 对复杂问题，优先使用列表、点号或其他结构化形式呈现答案。

---
### 背景资料：长期记忆库
这是从你的知识库中检索到的、与当前问题最相关的历史资料：
{retrieved_context}
---
### 背景资料：关于用户的工作偏好
这是你对用户【{user_name}】的专业领域和工作习惯的总结：
{professional_profile}
---
### 背景资料：近期对话上下文
这是我们最近的几轮对话：
{history_prompt}
---

【用户当前问题】:
{user_question}

【你的回答】:
"""
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    # 【可选优化】: 对于专业任务，可以适当降低一点点随机性
    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": history + [{"role": "user", "content": prompt}], "temperature": 0.4}
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"抱歉，连接AI服务时遇到了问题: {e}"

def reflect_and_update_memory(user_name, memory_file_path):
    with open(memory_file_path, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)
    user_history = memory_data.get(user_name, {}).get('history', {})
    if not user_history: return False
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if not user_history.get(today_str): return False
    
    conversation_text = "\n".join([f"用户: {t['query']}\nAI助手: {t['response']}" for t in user_history[today_str]])

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修改点 1/2：修改每日摘要Prompt ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    summary_prompt = f"""请以客观、精炼的语言，从以下对话中提取关键信息、待办事项(To-do)、或重要决策点。如果无特殊要点，则总结对话的核心议题。

对话内容:
{conversation_text}

今日工作要点总结:"""
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.1}
    
    made_update = False
    try:
        daily_summary = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload).json()['choices'][0]['message']['content'].strip()
        if 'summary' not in memory_data[user_name]: memory_data[user_name]['summary'] = {}
        memory_data[user_name]['summary'][today_str] = {"content": daily_summary}
        made_update = True
    except Exception as e:
        print(f"[错误] 生成每日摘要失败: {e}")

    all_summaries = "\n".join([f"日期 {d}: {s.get('content')}" for d, s in memory_data[user_name].get('summary', {}).items()])
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修改点 2/2：修改用户画像Prompt ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    portrait_prompt = f"""请根据以下长期的工作要点摘要，分析并总结用户【{user_name}】的专业领域、研究方向、工作习惯和核心关注点。输出一个简短、客观的第三人称工作画像描述。

工作要点历史:
{all_summaries}

更新后的用户工作画像:"""
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    payload['messages'][0]['content'] = portrait_prompt
    try:
        # 【修改】: 变量名从 overall_portrait 改为 professional_profile
        professional_profile = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload).json()['choices'][0]['message']['content'].strip()
        memory_data[user_name]['overall_personality'] = professional_profile # JSON中的字段名保持不变，避免破坏结构
        made_update = True
    except Exception as e:
        print(f"[错误] 更新全局画像失败: {e}")

    if made_update:
        with open(memory_file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
    return made_update
# ======================================================================================
# 3. 主程序入口
# ======================================================================================
if __name__ == "__main__":
    try:
        user_name = input("请输入您的名字: ").strip() or "DefaultUser"
        initialize_memory_file(MEMORY_FILE, user_name)
        
        # 使用我们新的、稳定的嵌入类
        embeddings = StableHuggingFaceEmbeddings(model_name_or_path=EMBEDDING_MODEL_NAME, device='cuda')
        
        db = load_and_index_memory(MEMORY_FILE, user_name, embeddings, force_rebuild=True)
        if not db: 
            print(f"[致命错误]: 无法为用户'{user_name}'加载或创建记忆索引。程序退出。")
            exit()
        
        conversation_history = []
        print(f"\n========================================\n欢迎回来, {user_name}！我是您的AI伴侣。\n输入 'exit' 或 'quit' 或 '退出' 来结束对话。\n========================================")
        while True:
            question = input(f"\n{user_name}: ").strip()
            if question.lower() in ["退出", "exit", "quit"]: break
            if not question: continue

            final_answer = answer_question(question, db, conversation_history, user_name)
            print(f"\n机器人: {final_answer}")

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": final_answer})
            if len(conversation_history) > HISTORY_WINDOW_SIZE * 2:
                conversation_history = conversation_history[-HISTORY_WINDOW_SIZE * 2:]

            with open(MEMORY_FILE, 'r+', encoding='utf-8') as f:
                memory_data = json.load(f)
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                if user_name not in memory_data: memory_data[user_name] = {"history": {}}
                if today_str not in memory_data[user_name].get('history', {}): memory_data[user_name]['history'][today_str] = []
                memory_data[user_name]['history'][today_str].append({"query": question, "response": final_answer})
                f.seek(0)
                json.dump(memory_data, f, ensure_ascii=False, indent=4)
                f.truncate()

            if reflect_and_update_memory(user_name, MEMORY_FILE):
                db = load_and_index_memory(MEMORY_FILE, user_name, embeddings, force_rebuild=True)

        print("机器人: 感谢使用，再见！")

    except Exception as e:
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!      程序遇到意外错误       !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        print("\n程序已终止。很抱歉出现这个问题。")