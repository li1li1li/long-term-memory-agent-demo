# state.py
import datetime
from memory import KnowledgeAndMemoryManager

class AppState:
    """封装单个用户的会话状态，持有记忆管理器实例。"""

    def __init__(self, user_id: str):
        print(f"--- 正在为用户 '{user_id}' 创建新的会话状态 ---")
        self.user_id = user_id
        # 持有统一的记忆管理器实例
        self.memory_manager = KnowledgeAndMemoryManager(user_id=self.user_id)
        # 线性对话历史，在内存中维护
        self.conversation_history: list[dict] = []

    def add_to_history(self, user_message: str, bot_message: str):
        """将会话添加到历史记录中。"""
        timestamp = datetime.datetime.now().isoformat()
        self.conversation_history.append({"role": "user", "content": user_message, "timestamp": timestamp})
        self.conversation_history.append({"role": "assistant", "content": bot_message, "timestamp": timestamp})

    def get_full_history_text(self) -> str:
        """获取格式化的完整对话历史文本。"""
        return "\n".join([f"{t['role']}: {t['content']}" for t in self.conversation_history])

    def get_today_history(self) -> list[dict]:
        """获取今天的对话历史。"""
        today = datetime.date.today()
        return [
            t for t in self.conversation_history 
            if 'timestamp' in t and datetime.datetime.fromisoformat(t['timestamp']).date() == today
        ]

    def clear_and_restart(self):
        """调用记忆管理器清除所有记忆，并重置自身状态。"""
        if self.memory_manager:
            self.memory_manager.clear_all_memory()
        self.conversation_history = []
        # 重新初始化管理器以获得一个干净的状态
        self.memory_manager = KnowledgeAndMemoryManager(user_id=self.user_id)
        print(f"--- 用户 '{self.user_id}' 的会话状态已重置 ---")