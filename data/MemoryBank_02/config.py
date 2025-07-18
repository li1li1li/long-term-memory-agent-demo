# config.py

# --- LLM 模型配置 ---
API_URL = "http://localhost:11434/v1/chat/completions"
API_KEY = "ollama"
MODEL_NAME = "qwen2:7b"

# --- 嵌入和重排模型配置 ---
EMBEDDING_MODEL_PATH = "../memoryBank-demo/models/all-MiniLM-L6-v2"
RERANKER_MODEL_PATH = '/root/autodl-tmp/memoryBank-demo/models/ms-marco-MiniLM-L-6-v2'

# ---co 文件和目录路径配置 ---
PROJECT_DIR = "./final_assistant_data/"
MEMORY_FILE = f"{PROJECT_DIR}/memory.json"
FAISS_INDEX_BASE_DIR = f"{PROJECT_DIR}/faiss_indices"

# --- 对话设置 ---
HISTORY_WINDOW_SIZE = 5