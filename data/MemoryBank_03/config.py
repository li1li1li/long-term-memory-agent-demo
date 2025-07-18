# config.py
import os

# LLM and Embedding Model Configuration
DEEPSEEK_API_URL = "http://localhost:11434/v1/chat/completions"
DEEPSEEK_API_KEY = "ollama"
DEEPSEEK_MODEL_NAME = "qwen2:7b"
EMBEDDING_MODEL_NAME = "../memoryBank-demo/models/all-MiniLM-L6-v2"
RERANKER_MODEL_PATH = '../memoryBank-demo/models/ms-marco-MiniLM-L-6-v2'

# Project Data Directories
PROJECT_DIR = "./final_assistant_data/"
MEMORY_FILE = os.path.join(PROJECT_DIR, "memory.json")
FAISS_INDEX_BASE_DIR = os.path.join(PROJECT_DIR, "faiss_indices")

# Assistant Behavior Configuration
HISTORY_WINDOW_SIZE = 5
SHORT_TERM_CACHE_CAPACITY = 10