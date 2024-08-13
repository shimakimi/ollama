import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# モデルとトークナイザーの準備
model_name = "meta-llama/Llama-2-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True
)

# 以下は変更なし
# LoRAの設定
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# ... (残りのコードは変更なし)