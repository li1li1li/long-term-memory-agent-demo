# config.py

# --- API 配置 ---
'''
DEEPSEEK_API_URL = "http://localhost:11434/v1/chat/completions"
DEEPSEEK_API_KEY = "ollama"  # API Key不再需要，可以填写任意字符作为占位符
DEEPSEEK_MODEL_NAME = "qwen2:1.5b" # 必须与你在Ollama中运行的模型名一致

'''


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# 在这里替换成你的真实 API Key
DEEPSEEK_API_KEY = "sk-***" 
DEEPSEEK_MODEL_NAME = "deepseek-chat"


# --- 模型配置 ---
EMBEDDING_MODEL_NAME = "./models/all-MiniLM-L6-v2"

# --- 文件路径配置 ---
PROJECT_DIR = "./memoryBankde-demo/"
KNOWLEDGE_BASE_FILE = f"{PROJECT_DIR}/my_knowledge.txt"
MEMORY_BANK_FILE = f"{PROJECT_DIR}/memory_bank.txt"
FAISS_INDEX_DIR = f"{PROJECT_DIR}/my_faiss_index"

# --- Gradio 对话配置 ---
HISTORY_WINDOW_SIZE = 3 # 短期上下文的窗口大小
