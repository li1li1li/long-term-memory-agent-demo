# app.py

import gradio as gr
import traceback
from typing import List, Tuple

# 导入我们自己的模块
import config
from embedding_utils import StableHuggingFaceEmbeddings
import memory_manager
import llm_integrations

# --- 全局加载区 ---
print("[应用启动]: 正在预加载嵌入模型...")
try:
    embeddings = StableHuggingFaceEmbeddings(model_name_or_path=config.EMBEDDING_MODEL_PATH, device='cuda')
    print("[应用启动]: 嵌入模型预加载成功。")
except Exception as e:
    print(f"[应用启动][致命错误]: 嵌入模型加载失败，程序无法启动。错误: {e}")
    exit()

def initialize_user(user_name: str) -> Tuple[str, List, dict, gr.update, gr.update]:
    """处理用户登录，初始化所有资源。"""
    if not user_name:
        gr.Warning("用户名不能为空！请输入您的名字。")
        return None, [], {}, gr.update(visible=True), gr.update(visible=False)
    print(f"[用户初始化]: 用户 '{user_name}' 正在登录...")
    memory_manager.initialize_memory_file(user_name)
    memory_manager.prune_memory(user_name)
    db = memory_manager.load_and_index_memory(user_name, embeddings, force_rebuild=True)
    if not db:
        gr.Error(f"无法为用户'{user_name}'加载或创建记忆索引。")
        return None, [], {}, gr.update(visible=True), gr.update(visible=False)
    print(f"[用户初始化]: 用户 '{user_name}' 准备就绪。")
    state = {"user_name": user_name, "db": db, "short_term_history": []}
    return f"欢迎, {user_name}！", [], state, gr.update(visible=False), gr.update(visible=True)

def chat_interface(message: str, chat_history: List[Tuple[str, str]], state: dict) -> Tuple[str, List, dict]:
    """聊天主函数，处理用户的每一次输入。"""
    user_name = state.get("user_name")
    db = state.get("db")
    short_term_history = state.get("short_term_history", [])
    if not user_name or not db:
        gr.Warning("请先输入您的名字并点击开始！")
        return "", chat_history, state
    
    chat_history.append((message, None))
    yield "", chat_history, state
    
    final_answer, used_chunks = llm_integrations.answer_question(message, db, short_term_history, user_name)
    memory_manager.reinforce_memory(user_name, used_chunks)
    memory_manager.save_conversation_turn(user_name, message, final_answer)
    
    short_term_history.append({"role": "user", "content": message})
    short_term_history.append({"role": "assistant", "content": final_answer})
    if len(short_term_history) > config.HISTORY_WINDOW_SIZE * 2:
        state["short_term_history"] = short_term_history[-config.HISTORY_WINDOW_SIZE * 2:]
    else:
        state["short_term_history"] = short_term_history
        
    if llm_integrations.reflect_and_update_memory(user_name):
        print("\n[主程序][系统]: 检测到摘要/画像更新，正在后台重建知识索引...")
        state["db"] = memory_manager.load_and_index_memory(user_name, embeddings, force_rebuild=True)

    chat_history[-1] = (message, final_answer)
    yield "", chat_history, state

with gr.Blocks(theme=gr.themes.Soft(), title="高级AI知识助手") as demo:
    state = gr.State({})
    gr.Markdown("# 高级AI知识助手 (Advanced AI Knowledge Assistant)")
    
    with gr.Row(visible=True) as login_view:
        with gr.Column(scale=3): name_input = gr.Textbox(label="请输入您的名字", placeholder="例如：张三")
        with gr.Column(scale=1): login_button = gr.Button("开始使用", variant="primary")
            
    with gr.Column(visible=False) as chat_view:
        welcome_banner = gr.Markdown("欢迎！")
        chatbot = gr.Chatbot(label="对话窗口", height=600)
        with gr.Row():
            msg_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...", show_label=False, scale=4, container=False)
            send_button = gr.Button("发送", variant="primary", scale=1)
    
    login_button.click(fn=initialize_user, inputs=[name_input], outputs=[welcome_banner, chatbot, state, login_view, chat_view])
    msg_input.submit(fn=chat_interface, inputs=[msg_input, chatbot, state], outputs=[msg_input, chatbot, state])
    send_button.click(fn=chat_interface, inputs=[msg_input, chatbot, state], outputs=[msg_input, chatbot, state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)