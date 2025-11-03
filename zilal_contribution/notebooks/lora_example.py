from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# from peft import LoraConfig, get_peft_model
# import torch

# base = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# tok = AutoTokenizer.from_pretrained(base)
# model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")

# lora_cfg = LoraConfig(
#     r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
# )
# model = get_peft_model(model, lora_cfg)  # << only adapters are trainable
# model.print_trainable_parameters()
