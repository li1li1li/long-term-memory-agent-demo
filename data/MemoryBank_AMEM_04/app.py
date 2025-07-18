import gradio as gr
import os
import datetime
from assistant.memory import StableHuggingFaceEmbeddings
from assistant.state import UserState
from assistant.logic import (
    generate_conversational_response,
    process_user_input_for_facts,
    reinforce_memory,
    reflect_and_update_memory
)
from config import EMBEDDING_MODEL_NAME

# --- 1. 初始化应用级的资源 ---
def load_embedding_model():
    print("[系统初始化]: 正在加载嵌入模型...")
    return StableHuggingFaceEmbeddings(model_name_or_path=EMBEDDING_MODEL_NAME)

embedding_model = load_embedding_model()


# --- 2. 定义Gradio的交互函数 ---
def handle_chat_interaction(message: str, chat_history: list, user_state: UserState, deep_mode: bool):
    """处理一次完整的聊天交互，并在最后统一提交变更。"""
    if not user_state:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "错误：请先在侧边栏输入您的名字并加载。"})
        yield "", chat_history, gr.update(value=user_state)
        return

    # 1. 更新聊天历史界面
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "..."})
    yield "", chat_history, gr.update(value=user_state)

    # 2. 调用 LLM 生成回答
    final_answer, used_chunks = generate_conversational_response(
        user_question=message,
        user_state=user_state,
        deep_mode=deep_mode
    )
    
    # 3. 更新聊天历史界面中的 AI 回复
    chat_history[-1]["content"] = final_answer
    yield "", chat_history, gr.update(value=user_state)
    
    # 4. 后台处理记忆和用户状态 (所有函数现在只在内存中操作)
    user_state.conversation_history.append({"role": "user", "content": message, "timestamp": datetime.datetime.now().isoformat()})
    user_state.conversation_history.append({"role": "assistant", "content": final_answer, "timestamp": datetime.datetime.now().isoformat()})
    
    user_state.short_term_cache.add(message, final_answer)

    if used_chunks:
        reinforce_memory(user_state.user_name, used_chunks, user_state)

    process_user_input_for_facts(user_state.user_name, message, user_state)

    reflect_and_update_memory(user_state.user_name, user_state)

    # 5. 【核心修改】最后，统一提交所有可能发生的内存变更
    # 这个方法会智能地检查是否有实际变更，然后才执行保存和重建操作。
    user_state.commit_changes()


def load_user(user_name: str):
    if not user_name or not user_name.strip():
        return None, [{"role": "assistant", "content": "请输入一个有效的用户名。"}], gr.update(interactive=False)
    
    state = UserState(user_name.strip(), embedding_model)
    return state, [{"role": "assistant", "content": f"你好，{user_name.strip()}！我已经准备好了。"}], gr.update(interactive=True, placeholder=f"向{user_name.strip()}提问...")

def clear_session(user_state: UserState):
    if user_state:
        user_state.clear_memory_and_restart() 
    return None, [], "请先在左侧加载用户...", gr.update(interactive=False)

# --- 3. 构建Gradio界面 ---
with gr.Blocks(theme=gr.themes.Soft(), title="全能AI助手 Demo") as demo:
    user_state = gr.State(None)
    
    gr.Markdown("# 全能AI助手 Demo (A-Mem精简版)")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### 控制面板")
            user_name_input = gr.Textbox(label="您的名字", placeholder="例如：zhangsan", info="输入名字后按回车加载您的专属记忆。")
            deep_mode_toggle = gr.Checkbox(label="深度思考模式", info="回答更深入，但响应更慢。")
            clear_button = gr.Button("清除记忆并重启", variant="stop")
        
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="对话窗口", height=600, type="messages", avatar_images=("./user_avatar.png", "./assistant_avatar.png"))
            msg_input = gr.Textbox(label="输入框", placeholder="请先在左侧加载用户...", interactive=False, scale=4)
    
    user_name_input.submit(
        fn=load_user, 
        inputs=[user_name_input], 
        outputs=[user_state, chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=handle_chat_interaction, 
        inputs=[msg_input, chatbot, user_state, deep_mode_toggle], 
        outputs=[msg_input, chatbot, user_state],
        queue=True
    )
    
    clear_button.click(
        fn=clear_session, 
        inputs=[user_state], 
        outputs=[user_state, chatbot, msg_input]
    )

# --- 4. 启动应用 ---
if __name__ == "__main__":
    demo.launch()