import sys
import os

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Remove one dirname call to point to Panacea-R1
# Add the project root to Python path
sys.path.insert(0, project_root)  # This will now add Panacea-R1 to the path


import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
from collections import deque
from threading import Lock
import time

from verl.utils.apis.pubmed import PubmedAPI
from verl.utils.apis.ctgov import CTGovAPI

# Add these at the top with other global variables
_request_times = deque(maxlen=20)  # Track last 20 requests
_request_lock = Lock()  # Thread-safe lock for request tracking

# CACHE_DIR = "/srv/local/data/linjc/hub"

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto',)
    model.eval()
    return tokenizer, model


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, processed_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str
    
    
def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_jsons = re.findall(pattern, text)
    
    if matched_jsons:
        extracted_json = matched_jsons[-1]  # get the final one
        return json.loads(extracted_json)
    else:
        # backup plan
        pattern = r"\{.*?\}"
        matched_jsons = re.findall(pattern, text, re.DOTALL)
        
        if matched_jsons:
            extracted_json = matched_jsons[-1]  # get the final one
            return json.loads(extracted_json)
        else:
            raise ValueError('No JSON structure found.')
        
def run_search_ctgov(search_query, search_api):
    pass

def run_search_pubmed(search_query, search_api, pub_date):
    # Rate limit checking
    current_time = time.time()
    with _request_lock:
        # Remove requests older than 1 second
        while _request_times and current_time - _request_times[0] > 1.0:
            _request_times.popleft()
        
        # Check if we're exceeding rate limit (10 requests per second)
        if len(_request_times) >= 10:
            print("\033[93m[Warning] PubMed rate limit (10 req/s) reached! Consider reducing batch size.\033[0m")
        
        # Record this request
        _request_times.append(current_time)
    
    # add date
    date_query_part = f'&datetype=pdat&mindate=1970/01/01&maxdate={pub_date}'
    search_query += date_query_part
    
    print('Query:', search_query)
    # search
    pmid_list = search_api.search_with_keywords(search_query, topk=-1)
    
    return pmid_list


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=8, search_api=None):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    pub_dates = df['pub_date'].tolist()
    
    model.to(device)
    generated_texts = {}
    error_count = 0
    recalls = []
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # try:
                output_ids = model.generate(**tokenized_inputs, max_new_tokens=512)
            # except:
                # continue
        
        for i, output in enumerate(output_ids):
            try:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                idx = batch_start + i
                # convert target from ndarray to list
                
                extracted_solution, processed_str = extract_solution(generated_text)
                query = json.loads(extracted_solution)['query']
                
                
                target = targets[idx].tolist()
                pub_date = pub_dates[idx]
                searched_pmids = run_search_pubmed(query, search_api, pub_date)
                
                hit_pmids = set(searched_pmids) & set(target)
                recall = len(hit_pmids) / len(target)          
                recalls.append(recall)
                
                generated_texts[idx] = {
                    "reasoning": processed_str,
                    "generated_query": query,
                    "pmid_list": searched_pmids,
                    "target": target,
                    "recall": recall
                }
                
                # print("Query: ", query)
                print("Recall: ", recall)
            except:
                print("Error: ", generated_text)
                error_count += 1
                continue
            
        time.sleep(1)
        
        print("Error count: ", error_count)
        print("Average recall: ", sum(recalls) / len(recalls))
        
        
        with open(os.path.join(save_dir, f"eval_results_{model_name}.json"), "w") as f:
            json.dump(generated_texts, f, indent=4)
    
    with open(os.path.join(save_dir, f"eval_results_{model_name}.json"), "w") as f:
        json.dump(generated_texts, f, indent=4)
    
    print("Error count: ", error_count)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/shared/eng/pj20/lmr_model/literature_search_3b/actor/global_step_1200")
    parser.add_argument("--data_path", type=str, default="data/literature_mining/base/test_full.parquet")
    parser.add_argument("--model_name", type=str, default="matching-qwen2.5-3b-inst-ppo-2gpus")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--api", type=str, default="pubmed")
    args = parser.parse_args()

    if args.api == "pubmed":
        if os.path.exists('verl/utils/reward_score/apis/pubmed_api.key'):
            api_key = open('verl/utils/reward_score/apis/pubmed_api.key', 'r').read().strip()
            search_api = PubmedAPI(api_key=api_key)
    elif args.api == "ctgov":
        search_api = CTGovAPI()
    else:
        raise ValueError(f"Invalid API: {args.api}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size, search_api)

if __name__ == "__main__":
    main()
