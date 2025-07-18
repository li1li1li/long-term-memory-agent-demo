import os 
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


## ======================================================================================
# 1. 配置区域: 在这里修改你的个人设置
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = "sk-a7ad78960dcf4f338385e20cd59534cb"  # <--- 在这里替换成你的真实 API Key
DEEPSEEK_MODEL_NAME = "deepseek-chat"


# 本地向量化模型的配置 (这个模型会自动从Hugging Face下载并运行在你的GPU上)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# AutoDL持久化存储路径配置
PROJECT_DIR = "/root/memoryBank-demo/"  # 你的项目主目录
KNOWLEDGE_BASE_FILE = os.path.join(PROJECT_DIR, "my_knowledge.txt")
FAISS_INDEX_DIR = os.path.join(PROJECT_DIR, "my_faiss_index")



# 2. 核心功能函数
# ======================================================================================

def build_knowledge_base():
    """
    【离线功能】读取知识库文件，将其处理成向量并构建FAISS索引。
    这个函数只在索引不存在时执行一次。
    """
    if os.path.exists(FAISS_INDEX_DIR):
        print(f"知识库索引 '{FAISS_INDEX_DIR}' 已存在，跳过构建过程。")
        return

    # 如果知识库文件不存在，则创建一个示例文件
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"未找到知识库文件，正在创建示例文件: {KNOWLEDGE_BASE_FILE}")
        os.makedirs(PROJECT_DIR, exist_ok=True) # 确保目录存在
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            f.write("RAG是检索增强生成的缩写，它先从知识库检索信息，再交给大模型生成答案。\n")
            f.write("FAISS是一个用于高效向量搜索的库，由Facebook AI开发。\n")
            f.write("在AutoDL平台上，为了利用GPU资源，应该安装faiss-gpu版本。\n")

    # 1. 读取文本文档
    print("正在读取知识库文件...")
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        knowledge_text = f.read()

    # 2. 将文本分割成小块 (Chunking)
    print("正在进行文本分割...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    text_chunks = text_splitter.split_text(knowledge_text)
    print(f"文本被分割成 {len(text_chunks)} 个小块。")

    # 3. 文本向量化 (Embedding) - 利用AutoDL的GPU
    print("正在进行文本向量化 (首次运行会下载模型，请稍候)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}  # 明确指定使用GPU
    )

    # 4. 构建并保存 FAISS 索引
    print("正在构建并保存FAISS索引...")
    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    print(f"知识库索引构建完成！已保存至 '{FAISS_INDEX_DIR}'")


def answer_question(user_question):
    """
    【在线功能】接收用户问题，从知识库检索，并调用DeepSeek API获取答案。
    """
    # 1. 加载本地的 FAISS 索引和向量化模型
    print("正在加载本地知识库...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # 2. 在知识库中进行相似度搜索 (利用GPU)
    print("正在检索相关信息...")
    retrieved_chunks = db.similarity_search(user_question, k=3)  # 检索最相关的3个文本块

    # 3. 准备上下文和构建 Prompt
    if not retrieved_chunks:
        print("未在知识库中检索到相关信息。")
        context = "无相关信息"
    else:
        context = "\n---\n".join([chunk.page_content for chunk in retrieved_chunks])

    prompt = f"""
你是一个专业的问答机器人。请严格根据下面提供的【参考资料】来回答【用户的问题】。
如果【参考资料】中没有相关信息，就直接回答："抱歉，根据我手头的资料，无法回答您的问题。"

【参考资料】:
{context}

【用户的问题】:
{user_question}

【你的回答】:
"""

    # 4. 调用 DeepSeek API
    print("正在请求 DeepSeek API...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": DEEPSEEK_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        api_result = response.json()
        return api_result['choices'][0]['message']['content']
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return "抱歉，请求AI服务时遇到了问题。"


# ======================================================================================
# 3. 主程序入口
# ======================================================================================

if __name__ == "__main__":
    # 程序启动时，首先确保知识库索引已经构建好
    build_knowledge_base()

    print("\n\n========================================")
    print("欢迎使用从零开始的问答机器人！")
    print("========================================")

    # 进入一个无限循环，接收用户输入并回答
    while True:
        question = input("\n请输入您的问题 (输入 '退出' 来结束程序): ")
        if question.strip().lower() in ["退出", "exit", "quit"]:
            print("感谢使用，再见！")
            break
        if not question.strip():
            continue

        final_answer = answer_question(question)
        print("\n[机器人的回答]:")
        print(final_answer)
