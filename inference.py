import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel, PeftConfig

# Configuration
base_model_name = "mistralai/Mistral-7B-v0.1"
# Path to the saved adapter (from train.py)
adapter_path = "mistral-7b-finetuned-custom" 

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the adapter
print(f"Loading adapter from {adapter_path}...")
model = PeftModel.from_pretrained(model, adapter_path)

# Merge if you want to export the full model, but for inference we can keep it usually.
# model = model.merge_and_unload() 

# Inference pipeline
pipe = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=200
)

# Test prompt
prompt = "What is the capital of France?"
formatted_prompt = f"<s>[INST] {prompt} [/INST]"

# Generate
print("Generating...")
result = pipe(formatted_prompt)
print(result[0]['generated_text'])
