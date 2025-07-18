# app.py
import gradio as gr
import state
import logic

# --- Gradio 交互函数 ---

def load_user(user_name: str):
    """根据用户名加载或创建用户状态。"""
    if not user_name or not user_name.strip():
        gr.Warning("请输入一个有效的用户名。")
        return None, [], gr.update(interactive=False)
    
    # 直接创建 AppState 实例
    user_state_instance = state.AppState(user_name.strip())
    
    return user_state_instance, [{"role": "assistant", "content": f"你好，{user_name.strip()}！我已经准备好了。"}], gr.update(interactive=True, placeholder=f"向{user_name.strip()}提问...")

def handle_chat_interaction(message: str, chat_history_for_display: list, user_state: state.AppState, deep_mode: bool):
    """处理一次完整的聊天交互。"""
    if not user_state:
        # 这种情况理论上不应该发生，因为输入框在加载用户前是禁用的
        gr.Error("错误：用户状态未加载。请先加载用户。")
        return "", chat_history_for_display, user_state

    # 更新UI以显示用户消息和AI思考占位符
    chat_history_for_display.append({"role": "user", "content": message})
    chat_history_for_display.append({"role": "assistant", "content": "🤔..."})
    yield "", chat_history_for_display, user_state

    # 1. 生成回答
    final_answer, used_docs = logic.generate_conversational_response(message, user_state, deep_mode)
    chat_history_for_display[-1] = {"role": "assistant", "content": final_answer}
    yield "", chat_history_for_display, user_state
    
    # 2. 后台处理记忆（更新历史、强化、反思等）
    user_state.add_to_history(message, final_answer)
    
    if used_docs:
        user_state.memory_manager.reinforce_memory(used_docs)

    logic.process_user_input_for_facts(user_state.user_id, message, user_state)
    
    logic.reflect_and_update_memory(user_state.user_id, user_state)

def clear_session(user_state: state.AppState):
    """清除当前用户的记忆和会话。"""
    if user_state:
        user_state.clear_and_restart()
    return None, [], "请先在左侧加载用户...", gr.update(interactive=False)

# --- 构建Gradio界面 (与您提供的版本几乎一致) ---
with gr.Blocks(theme=gr.themes.Soft(), title="全能AI助手 (最终版)") as demo:
    # Gradio State 用于在会话中持久化我们的 AppState 对象
    gr_user_state = gr.State(None) 
    
    gr.Markdown("# 全能AI助手 (知识图谱 & 深度思考最终版)")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            # ... (UI组件与您提供的 app.py 完全相同)
            user_name_input = gr.Textbox(label="您的名字", placeholder="例如：zhangsan", info="输入名字后按回车加载您的专属记忆。")
            deep_mode_toggle = gr.Checkbox(label="深度思考模式", info="回答更深入，但响应更慢。")
            clear_button = gr.Button("清除记忆并重启", variant="stop")
        
        
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="对话窗口", height=600, type="messages", avatar_images=("./user_avatar.png", "./assistant_avatar.png"))
            msg_input = gr.Textbox(label="输入框", placeholder="请先在左侧加载用户...", interactive=False, scale=4)
    
    # 事件绑定
    user_name_input.submit(
        fn=load_user, 
        inputs=[user_name_input], 
        outputs=[gr_user_state, chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=handle_chat_interaction, 
        inputs=[msg_input, chatbot, gr_user_state, deep_mode_toggle], 
        outputs=[msg_input, chatbot, gr_user_state]
    )
    
    clear_button.click(
        fn=clear_session, 
        inputs=[gr_user_state], 
        outputs=[gr_user_state, chatbot, msg_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)