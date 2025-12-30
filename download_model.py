'''
# download_model.py（仅用于下载，下载完成后可删除）
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "Qwen/Qwen2.5-0.5B"
# 本地保存路径（可自定义，例如 "./local-distilgpt2"）
local_model_path = "./local-model"

# 下载并保存模型和tokenizer
print(f"开始下载模型 {model_name} 到 {local_model_path}...")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
print("下载完成！")
'''

import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi

# --- 配置区 ---
# 1. 你的微调模型 (权重从这里下)
MY_MODEL_ID = "Xunden/Qwen2.5-7B-GPU-Expert"

# 2. 官方底座模型 (分词器从这里借，确保 100% 不报错)
OFFICIAL_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# 3. 保存路径
LOCAL_DIR = "./qwen-2.5-merged-final"

# 4. 你的 Token
MY_TOKEN = "ms-764d30ab-b680-4ec4-a663-eee471c2df96" 

def download():
    # --- 步骤 1: 登录 ---
    try:
        print("--- [1/3] 正在登录 ModelScope ---")
        api = HubApi()
        api.login(MY_TOKEN)
    except Exception as e:
        print(f"登录提示: {e}")

    # --- 步骤 2: 下载你的微调模型 (主要是为了权重) ---
    print(f"--- [2/3] 下载微调模型权重: {MY_MODEL_ID} ---")
    try:
        snapshot_download(
            MY_MODEL_ID,
            local_dir=LOCAL_DIR,
            revision='master',
            # 忽略掉可能损坏的分词器文件，稍后用官方的补
            ignore_file_pattern=[".git", "README.md", "tokenizer.json", "vocab.json", "merges.txt"]
        )
    except Exception as e:
        print(f"警告：模型下载遇到问题，尝试继续... {e}")
        # 这里不 raise，因为可能部分文件已下载

    # --- 步骤 3: 下载官方分词器并覆盖 (双保险) ---
    print(f"--- [3/3] 正在从官方仓库 {OFFICIAL_MODEL_ID} 修复分词器 ---")
    # 只下载分词器相关文件
    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
    
    snapshot_download(
        OFFICIAL_MODEL_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=tokenizer_files # 只下这几个文件
    )
    
    # --- 最终校验 ---
    if os.path.exists(os.path.join(LOCAL_DIR, "model.safetensors.index.json")) and \
       os.path.exists(os.path.join(LOCAL_DIR, "tokenizer.json")):
        print("\n✅ 下载与修复完成！模型与分词器已就绪。")
    else:
        raise RuntimeError("❌ 下载验证失败：关键文件缺失！")

if __name__ == "__main__":
    download()