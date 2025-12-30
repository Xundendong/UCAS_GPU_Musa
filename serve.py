import torch
# 尝试导入摩尔线程专用库 (有些环境需要显式导入，有些不需要，为了保险加上)
try:
    import torch_musa
except ImportError:
    pass

from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn

# 设置离线模式，防止 transformers 尝试联网
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 修改模型路径 ---
# 必须指向 download_model.py 下载的那个文件夹
LOCAL_MODEL_PATH = "./qwen-2.5-merged-final"

print(f"正在加载模型: {LOCAL_MODEL_PATH} ...")

# --- 2. 加载 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH, 
    trust_remote_code=True
)

# --- 3. 加载模型 (针对摩尔线程优化) ---
print("正在加载权重到 MUSA 设备...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 关键：使用半精度，速度快且省显存
    device_map="musa"           # 关键：指定设备为摩尔线程 GPU (musa)
)

# 将模型设置为评估模式
model.eval()

# --- API 定义 ---
app = FastAPI(title="MUSA Inference Server")

class PromptRequest(BaseModel):
    prompt: str

class PredictResponse(BaseModel):
    response: str

@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest):
    """
    摩尔线程专用推理接口
    """
    # --- 4. 构建 Qwen 专用的 Chat 模板 ---
    # Qwen2.5 需要这种格式才能发挥出微调的效果
    prompt = request.prompt
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": prompt}
    ]
    
    # 使用 tokenizer 自动应用模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to("musa")

    # --- 5. 执行推理 ---
    with torch.no_grad(): # 这一步很重要，不计算梯度能省很多显存
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,       # 最大生成长度
            do_sample=False,          # 贪婪搜索，结果稳定
            temperature=0.01,         # 只有 do_sample=True 时才生效，这里设低点防止乱码
            pad_token_id=tokenizer.eos_token_id
        )

    # --- 6. 提取输出 ---
    # 去掉输入部分的 token，只保留新生成的
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return PredictResponse(response=response_text.strip())

@app.get("/")
def health_check():
    return {"status": "ok"}

# 如果你需要本地测试，可以使用: python serve.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)