import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json

tqdm.pandas()

sys.path.append('./')

from src.utils.gpt_azure import gpt_chat_4o, gpt_chat_4omini



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument("--data_path", type=str, default="data/matching/qwen-instruct/test.parquet")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    

    df = pd.read_parquet(args.data_path)

    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()

    i = 0
    output_dict = {}
    for idx, prompt in enumerate(tqdm(inputs)):
        pdb.set_trace()
        # extract the prompt from "<|im_start|>user" to <|im_start|>assistant
        prompt = prompt.split("<|im_start|>user", 1)[1]
        prompt = prompt.split("<|im_start|>assistant", 1)[0]

        prompt = prompt.replace("<im_end>", "")
        prompt = prompt.strip()

        prompt = prompt.replace("{", "(")
        prompt = prompt.replace("}", ")")
        if args.model_name == 'gpt-4o-mini':
            
                decoded = gpt_chat_4omini(prompt)
        elif args.model_name == 'gpt-4o':
            
                decoded = gpt_chat_4o(prompt)
        
        output_dict[idx] = {
            'output': decoded,
            'label': targets[idx]
        }
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)

        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)