import torch
# 尝试导入摩尔线程专用库 (必须导入)
try:
    import torch_musa
except ImportError:
    pass

from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 引入 vLLM
from vllm import LLM, SamplingParams
# 仍需 Tokenizer 来处理对话模板
from transformers import AutoTokenizer

# --- 1. 修改模型路径 ---
LOCAL_MODEL_PATH = "./qwen-2.5-merged-final"

print(f"正在加载模型 (vLLM): {LOCAL_MODEL_PATH} ...")

# --- 2. 加载 Tokenizer (仅用于处理 Chat 模板) ---
# vLLM 内部其实也有 tokenizer，但为了精确控制 Prompt 格式，
# 我们在外部处理好模板再喂给 vLLM 是最稳妥的。
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH, 
    trust_remote_code=True
)

# --- 3. 加载 vLLM 模型 (针对摩尔线程优化) ---
# gpu_memory_utilization=0.9 表示占用 90% 显存
# 如果你运行报错 OOM (Out Of Memory)，请尝试改成 0.8 或 0.7
print("正在初始化 vLLM 引擎...")
llm = LLM(
    model=LOCAL_MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,      # 单卡模式
    gpu_memory_utilization=0.9,  # 显存占用比例
    dtype="float16",              # 强制半精度
    device="musa"
)

# --- API 定义 ---
app = FastAPI(title="MUSA vLLM Inference Server")

class PromptRequest(BaseModel):
    prompt: str

class PredictResponse(BaseModel):
    response: str

@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest):
    """
    基于 vLLM 的摩尔线程推理接口
    """
    prompt = request.prompt
    
    # --- 4. 构建 Prompt (应用 Chat 模板) ---
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": prompt}
    ]
    
    # 将对话列表转换为纯文本 prompt
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # --- 5. 设置采样参数 ---
    # vLLM 的参数是在这里设置的
    sampling_params = SamplingParams(
        temperature=0.7,       # 控制随机性
        top_p=0.8,
        max_tokens=512,        # 最大生成长度
        stop_token_ids=[tokenizer.eos_token_id] # 遇到结束符停止
    )

    # --- 6. 执行推理 (vLLM) ---
    # vLLM 接受列表输入，这里我们放入一个请求
    outputs = llm.generate([text_input], sampling_params)

    # --- 7. 提取输出 ---
    # vLLM 的 output 对象包含了原始 prompt 和生成的 text
    generated_text = outputs[0].outputs[0].text
    
    return PredictResponse(response=generated_text.strip())

@app.get("/")
def health_check():
    return {"status": "ok", "backend": "vllm_musa"}

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)