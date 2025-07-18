# app.py

import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings

# 导入我们的配置和核心逻辑
import config
from core_logic import build_knowledge_base, load_vector_db, answer_question, extract_and_save_memory

# --- 1. 全局对象加载 (程序启动时执行一次) ---
print("[系统启动] 正在初始化...")
build_knowledge_base()
with open(config.MEMORY_BANK_FILE, "r", encoding="utf-8") as f:
    known_memories = set(line.strip() for line in f if line.strip())

print("[系统启动] 正在加载向量化模型...")
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cuda'})
db = load_vector_db(embeddings)
print("[系统启动] 初始化完成！Gradio界面即将启动。")


# --- 2. Gradio 核心逻辑 ---
def chat_response(message, chat_history):
    global db, known_memories, embeddings # 声明我们需要修改全局的db和known_memories
    
    conversation_history = []
    for user_msg, assistant_msg in chat_history:
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": assistant_msg})
    
    final_answer = answer_question(message, db, conversation_history)
    
    current_turn = {"user": message, "assistant": final_answer}
    if extract_and_save_memory(current_turn, known_memories):
        print("[Gradio后台：检测到记忆更新，正在重建知识库...]")
        build_knowledge_base(force_rebuild=True)
        db = load_vector_db(embeddings) # 重新加载更新后的数据库
        print("[Gradio后台：知识库重建完成！]")
        
    return final_answer

# --- 3. Gradio 界面定义 ---
with gr.Blocks(theme=gr.themes.Soft(), title="RAG-Bot with Memory") as demo:
    gr.Markdown("# 🧠 带记忆的AI聊天伙伴")
    gr.Markdown("这是一个结合了`RAG`、`长期记忆`和`短期上下文`的AI聊天机器人Demo。")
    
    chatbot = gr.Chatbot(
        label="聊天窗口",
        bubble_full_width=False,
        avatar_images=(None, "https://api.deepseek.com/deepseek.png"),
        height=600
    )
    
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="你好，可以问我任何问题...", lines=2, scale=4)
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    def respond(message, chat_history):
        bot_message = chat_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    submit_btn.click(respond, [txt, chatbot], [txt, chatbot])
    txt.submit(respond, [txt, chatbot], [txt, chatbot])

# --- 4. 启动Gradio服务器 ---
demo.launch(server_name="0.0.0.0")
