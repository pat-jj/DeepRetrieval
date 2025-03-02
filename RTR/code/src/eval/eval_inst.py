import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os

CACHE_DIR = "/srv/local/data/linjc/hub"

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto',)
    model.eval()
    return tokenizer, model


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=8):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    
    model.to(device)
    generated_texts = {}
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # try:
                output_ids = model.generate(**tokenized_inputs, max_new_tokens=1024)
            # except:
                # continue
        
        for i, output in enumerate(output_ids):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            idx = batch_start + i
            generated_texts[idx] = {
                "generated_text": generated_text,
                "target": targets[idx]
            }
        
        with open(os.path.join(save_dir, f"eval_results_{model_name}.json"), "w") as f:
            json.dump(generated_texts, f, indent=4)
    
    with open(os.path.join(save_dir, f"eval_results_{model_name}.json"), "w") as f:
        json.dump(generated_texts, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/Panacea-Zero/matching-qwen2.5-3b-inst-ppo-2gpus/actor/global_step_400")
    parser.add_argument("--data_path", type=str, default="data/matching/qwen-instruct/test.parquet")
    parser.add_argument("--model_name", type=str, default="matching-qwen2.5-3b-inst-ppo-2gpus")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size)

if __name__ == "__main__":
    main()
