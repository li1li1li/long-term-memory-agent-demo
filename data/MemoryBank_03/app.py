import gradio as gr
from assistant.memory import StableHuggingFaceEmbeddings
from assistant.state import UserState
from assistant.logic import (
    generate_conversational_response,
    answer_question_deep_thought,
    extract_and_store_facts,
    reinforce_memory,
    load_and_index_memory,
    reflect_and_update_memory
)
from config import MEMORY_FILE, EMBEDDING_MODEL_NAME

# --- 1. 初始化应用级的资源 ---
embedding_model = StableHuggingFaceEmbeddings(model_name_or_path=EMBEDDING_MODEL_NAME)

# --- 2. 定义Gradio的交互函数 (最终版) ---
def handle_chat_interaction(message: str, chat_history: list, user_state: UserState, deep_mode: bool):
    """(最终版)处理聊天交互，使用统一的“核心人格”模型。"""
    if not user_state:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "错误：请先在侧边栏输入您的名字并加载。"})
        return "", chat_history, user_state

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""}) # 占位
    yield "", chat_history, user_state

    final_answer = ""
    used_chunks = []

    # --- 核心改动：根据“深度思考”开关，选择不同的应答模式 ---
    if deep_mode:
        # 深度模式：调用专门的、复杂的分析函数
        print("[Gradio]: 已激活【深度思考模式】")
        final_answer, used_chunks = answer_question_deep_thought(message, user_state)
    else:
        # 标准模式：调用全新的、统一的“核心人格”应答函数
        print("[Gradio]: 已激活【标准模式】")
        final_answer, used_chunks = generate_conversational_response(message, user_state)

    # --- 后台记忆更新 ---
    # 1. 事实提取现在变成一个通用的后台任务
    if extract_and_store_facts(user_state.user_name, MEMORY_FILE, message, final_answer):
        print("[Gradio]: 发现新事实，正在后台重建长期记忆索引...")
        user_state.long_term_db = load_and_index_memory(MEMORY_FILE, user_state.user_name, user_state.embeddings, force_rebuild=True)

    # 2. 记忆加固（只对使用了记忆的回答有效）
    if used_chunks:
        reinforce_memory(user_state.user_name, used_chunks, MEMORY_FILE)
    
    # 3. 每日反思（只在有足够对话后触发）
    if reflect_and_update_memory(user_state.user_name, MEMORY_FILE):
         print("[Gradio]: 反思机制触发，正在后台重建长期记忆索引...")
         user_state.long_term_db = load_and_index_memory(MEMORY_FILE, user_state.user_name, user_state.embeddings, force_rebuild=True)

    # 4. 更新线性历史和短期缓存
    user_state.conversation_history.append({"role": "user", "content": message})
    user_state.conversation_history.append({"role": "assistant", "content": final_answer})
    user_state.short_term_cache.add(message, final_answer)

    # 用最终答案更新UI
    chat_history[-1]["content"] = final_answer
    yield "", chat_history, user_state


def load_user(user_name: str):
    if not user_name or not user_name.strip():
        error_message = [{"role": "assistant", "content": "请输入一个有效的用户名。"}]
        return None, gr.update(value=error_message), gr.update(interactive=False)
    state = UserState(user_name.strip(), embedding_model)
    initial_message = [{"role": "assistant", "content": f"你好，{user_name}！我已经准备好了。"}]
    return state, gr.update(value=initial_message), gr.update(interactive=True, placeholder=f"向{user_name}提问...")

def clear_session(user_state: UserState):
    if user_state:
        user_state.clear_memory_and_restart()
    return None, [], "请先在左侧加载用户...", gr.update(interactive=False)

# --- 3. 构建Gradio界面 ---
with gr.Blocks(theme=gr.themes.Soft(), title="全能AI助手 Demo") as demo:
    user_state = gr.State(None)
    gr.Markdown("# 全能AI助手 Demo 🤖\n一个融合了双层记忆、事实提取和多模式推理的智能助理。")
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### 控制面板")
            user_name_input = gr.Textbox(label="您的名字", placeholder="例如：zhangsan", info="输入名字后按回车加载您的专属记忆。")
            deep_mode_toggle = gr.Checkbox(label="深度思考模式", info="回答更深入，但响应更慢。")
            clear_button = gr.Button("清除记忆并重启", variant="stop")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="对话窗口", height=600, type="messages", avatar_images=("./user_avatar.png", "./assistant_avatar.png"))
            msg_input = gr.Textbox(label="输入框", placeholder="请先在左侧加载用户...", interactive=False, scale=4)
    
    user_name_input.submit(fn=load_user, inputs=[user_name_input], outputs=[user_state, chatbot, msg_input])
    msg_input.submit(fn=handle_chat_interaction, inputs=[msg_input, chatbot, user_state, deep_mode_toggle], outputs=[msg_input, chatbot, user_state])
    clear_button.click(fn=clear_session, inputs=[user_state], outputs=[user_state, chatbot, msg_input])

# --- 4. 启动应用 ---
if __name__ == "__main__":
    demo.launch()