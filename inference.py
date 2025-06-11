# load & serve via FastAPI instead of CLI
import os
from peft import PeftModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from train import resize_embeddings

# uvicorn inference:app --host 0.0.0.0 --port 8000 --reload

# —– load config & pick latest checkpoint —–
base_model_id = os.getenv("BASE_MODEL_ID", "unsloth/Llama-3.2-3B-bnb-4bit")
model_dir     = os.getenv("LORA_WEIGHTS",   "outputs/checkpoint-5")

if os.path.isdir(model_dir):
    subs = sorted(
        d for d in os.listdir(model_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(model_dir, d))
    )
    if subs:
        model_dir = os.path.join(model_dir, subs[-1])

# —– tokenizer init (load from your fine-tuned folder so the new tokens & embeddings are preserved) —–
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# —– model init (load merged, fine-tuned weights + embeddings in one go) —–
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
    device_map="auto",
)

print(base_model.get_input_embeddings().weight.shape)
print(base_model.get_input_embeddings().weight[-4:])


# Load special embedding weights if they exist
if os.path.exists(os.path.join(model_dir, "added_embeddings.pt")):
    added_embeddings = torch.load(os.path.join(model_dir, "added_embeddings.pt"))
    resize_embeddings(base_model, tokenizer, added_embeddings)

else:
    # Resize embeddings to match tokenizer size
    resize_embeddings(base_model, tokenizer)

# Load the fine-tuned model weights
model = PeftModel.from_pretrained(base_model, model_dir)

print(base_model.get_input_embeddings().weight.shape)
print(base_model.get_input_embeddings().weight[-4:])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —– FastAPI app —–
app = FastAPI()

class CompletionQuery(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/generate/")
async def generate(query: CompletionQuery):
    inputs = tokenizer(query.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=query.max_new_tokens,
            temperature=query.temperature,
            top_k=query.top_k,
            top_p=query.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}
