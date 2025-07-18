#!/bin/bash

# --- 脚本设置：如果任何命令失败，则立即退出 ---
set -e

# --- 1. 创建一个全新的、隔离的conda环境，名为 "memory_env" ---
echo ">>> 正在创建新的Conda环境 'memory_env'..."
conda create -n memory_env python=3.10 -y

# --- 2. 激活这个新环境 ---
echo ">>> 正在激活 'memory_env'..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate memory_env

# --- 3. 在新环境中，安装一个已知可以稳定工作的、完整的库“全家桶” ---
echo ">>> 正在安装所有必需的库（这将需要一些时间）..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install langchain-community==0.0.31 faiss-cpu==1.7.4 sentence-transformers==2.2.2
pip install transformers==4.30.2 accelerate==0.21.0 huggingface-hub==0.17.3
pip install requests

echo ">>> 环境配置成功！"

# --- 4. 运行您的最终版Python程序 ---
echo ">>> 即将运行您的MemoryBank程序..."
cd /root/autodl-tmp/memoryBank-demo/
python memoryBank.py

