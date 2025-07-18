# config.py
import os

# --- LLM and Embedding Model Configuration ---
# 使用 Ollama

DEEPSEEK_API_URL = "http://localhost:11434/v1"
DEEPSEEK_API_KEY = "ollama"
DEEPSEEK_MODEL_NAME = "qwen2:7b" # 请确保已下载: `ollama pull qwen2:7b`
'''
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# 在这里替换成你的真实 API Key
DEEPSEEK_API_KEY = "sk-a7ad78960dcf4f338385e20cd59534cb" 
DEEPSEEK_MODEL_NAME = "deepseek-chat"
'''
# 本地嵌入和重排序模型路径
EMBEDDING_MODEL_NAME = "../memoryBank-demo/models/all-MiniLM-L6-v2"
RERANKER_MODEL_PATH = '../memoryBank-demo/models/ms-marco-MiniLM-L-6-v2'

# --- Neo4j Knowledge Graph Configuration ---
NEO4J_URL = "neo4j+s://1f668954.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "5cgG-6A69XMQmn0vA4raFzN0axk111ME-oiMg0_Y1lQ" # 【必须】替换为您自己的密码
NEO4J_DATABASE = "neo4j"

# --- Project Data Directories ---
# 用于存储FAISS索引等
PROJECT_DIR = "./final_assistant_data/"
FAISS_INDEX_BASE_DIR = os.path.join(PROJECT_DIR, "faiss_indices")
os.makedirs(FAISS_INDEX_BASE_DIR, exist_ok=True)

# --- Assistant Behavior Configuration ---
# 短期记忆容量
SHORT_TERM_CACHE_CAPACITY = 10
# 遗忘曲线的保留阈值
FORGETTING_THRESHOLD = 0.25