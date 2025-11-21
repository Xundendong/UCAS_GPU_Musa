import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import pipeline, set_seed

# --- 模型加载 ---
# 为了保证最小化和快速启动，我们使用 distilgpt2
# 这是GPT-2的一个更小、更快、更轻的版本
# 在生产或更复杂的场景中，可以替换为其他模型


# 本地模型路径（和步骤1中保存的路径一致）
LOCAL_MODEL_PATH = "./local-distilgpt2"

# --- 模型加载（从本地加载，无需网络）---
print(f"从本地加载模型：{LOCAL_MODEL_PATH}")
# 手动加载本地模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)
# 初始化pipeline（使用本地模型）
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    # device=0  # 若有GPU可启用
)
set_seed(42)

# --- API 定义 ---
# 创建FastAPI应用实例
app = FastAPI(
    title="Simple Inference Server",
    description="A simple API to run a small language model."
)

# 定义API请求的数据模型
class PromptRequest(BaseModel):
    prompt: str

# 定义API响应的数据模型
class PredictResponse(BaseModel):
    response: str
    
# --- API 端点 ---
@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest):
    """
    接收一个prompt，使用加载的模型进行推理，并返回结果。
    这是评测程序将会调用的主要端点。
    """
    # 学生可以在这里修改推理逻辑
    # 例如，调整max_length, num_return_sequences等参数
    model_output = generator(
        request.prompt, 
        max_length=50, 
        num_return_sequences=1
    )
    
    # 从模型输出中提取生成的文本
    generated_text = model_output[0]['generated_text']
    
    # 返回符合格式的响应
    return PredictResponse(response=generated_text)

@app.get("/")
def health_check():
    """
    健康检查端点，用于确认服务是否启动成功。
    """
    return {"status": "ok"}