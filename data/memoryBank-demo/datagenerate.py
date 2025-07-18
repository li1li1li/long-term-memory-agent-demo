# generate_data_with_deepseek.py

import json
import requests
import time
import os
import traceback

# ======================================================================================
# 1. 配置区域
# ======================================================================================

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ▼▼▼ 请务必在这里填入您自己的DeepSeek API Key ▼▼▼
DEEPSEEK_API_KEY = "sk-a7ad78960dcf4f338385e20cd59534cb" 
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL_TO_USE = "deepseek-chat"  # 使用DeepSeek的在线聊天模型

# 定义我们希望生成的新数据集的文件名和路径
# 将直接存放到LLaMA Factory的数据目录
OUTPUT_FILE = "../LLaMA-Factory/data/deepseek_generated_academic_data.json" 

# 您希望生成多少条问答数据
NUM_SAMPLES_TO_GENERATE = 100 # 建议先从50-100条开始，以控制成本和时间

# 这是我们给DeepSeek模型的“出题”指令
GENERATION_PROMPT_TEMPLATE = """
你是一位顶尖的AI数据生成专家，你的任务是为微调一个面向学术研究和办公场景的AI知识助手，创造高质量、多样化的问答对（instruction-output pairs）。

请严格遵循以下指令，生成一个符合要求的JSON对象。

### 指令:
1.  **角色**: AI知识助手，面向学术研究和办公场景。
2.  **风格**: 回答必须专业、严谨、逻辑清晰、结构化。对于复杂问题，应使用点列或编号来组织答案。
3.  **主题**: 生成的问题（instruction）应覆盖以下至少一个领域：
    * **学术研究**: 概念解释（如“置信区间”、“P值”）、研究方法论对比、论文结构建议、文献综述技巧。
    * **办公效率**: 复杂的任务规划（如“制定季度OKRs”）、专业的邮件撰写模板、项目复盘报告框架、高效会议纪要方法。
    * **数据分析**: 数据清洗步骤、特定分析方法（如“回归分析”）的应用场景、数据可视化图表选择。
    * **专业写作**: 比如 "如何撰写一封专业的商务邮件，以跟进一个重要的潜在客户？"
4.  **避免简单问题**: 不要生成“什么是Python？”这类过于简单或常识性的问题。问题应具有实际应用价值和一定的深度。
5.  **输出格式**: 你的回答必须是一个严格的JSON对象，只包含"instruction"、"input"（通常为空）和"output"三个键，不要有任何其他多余的文字、解释或代码块标记。

```json
{
  "instruction": "在这里填写一个具体的、有深度的问题或指令。",
  "input": "",
  "output": "在这里提供一个堪称典范的、高质量、详细且结构化的回答。"
}
现在，请为我生成一个全新的、与任何常见示例都不同的问答对。
"""

#======================================================================================
def generate_single_sample_with_deepseek():
    """调用DeepSeek API生成一个单独的问答对样本。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": MODEL_TO_USE,
        "messages": [{"role": "user", "content": GENERATION_PROMPT_TEMPLATE}],
        "temperature": 0.9,  # 提高随机性以获得更多样化的数据
        "max_tokens": 4096,
        "stream": False
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        content_str = response.json()['choices'][0]['message']['content']
        # 清理并解析JSON
        # 有时模型返回的JSON会被包裹在```json ... ```中，需要提取出来
        if "```json" in content_str:
            json_str = content_str.split("```json")[1].split("```")[0].strip()
        else: 
            json_str = content_str.strip()     
        return json.loads(json_str)
    except requests.exceptions.RequestException as e:
        print(f"    - [网络错误] 连接DeepSeek API失败: {e}")
        return None
    except json.JSONDecodeError:
        print(f"    - [格式错误] API返回的不是有效的JSON，内容: {content_str[:100]}...")
        return None
    except Exception as e:
        print(f"    - [未知错误] 生成样本时发生错误: {e}")
        if 'response' in locals():
            print(f"    - API返回内容: {response.text}")
        return None

def generate_dataset():
    """主函数，循环生成指定数量的数据并保存。"""
    # 检查API Key是否已填写
    if "sk-xxxxxxxx" in DEEPSEEK_API_KEY or not DEEPSEEK_API_KEY:
        print("错误：请先在脚本第21行填写您自己的DeepSeek API Key！")
        return
    
    all_data = []
    print(f"准备使用DeepSeek API生成 {NUM_SAMPLES_TO_GENERATE} 条合成数据...")
    retries = 0
    max_retries = 3

    while len(all_data) < NUM_SAMPLES_TO_GENERATE:
        print(f"正在生成第 {len(all_data) + 1} / {NUM_SAMPLES_TO_GENERATE} 条数据...")
        sample = generate_single_sample_with_deepseek()
        
        if sample and "instruction" in sample and "output" in sample:
            all_data.append(sample)
            print(f"    -> 生成成功: instruction = \"{sample['instruction'][:40]}...\"")
            retries = 0  # 成功后重置重试计数器
            time.sleep(2)  # 成功后等待2秒
        else:
            retries += 1
            print(f"    -> 生成失败或格式错误，将在10秒后进行第 {retries}/{max_retries} 次重试...")
            if retries >= max_retries:
                print("    -> 已达到最大重试次数，可能存在持续的网络问题或API Key问题，程序将终止。")
                break
            time.sleep(10)
    
    if not all_data:
        print("\n未能成功生成任何数据，请检查您的API Key和网络连接。")
        return
    
    try:
        # 确保目标目录存在
        output_dir = os.path.dirname(OUTPUT_FILE)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"\n数据集生成完毕！共 {len(all_data)} 条数据已成功保存到 '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"\n错误：写入输出文件 '{OUTPUT_FILE}' 失败。错误信息: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    generate_dataset()