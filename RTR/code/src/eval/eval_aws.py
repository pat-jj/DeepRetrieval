import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json

tqdm.pandas()

sys.path.append('./')

from src.utils.claude_aws import chat_haiku, chat_sonnet



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='claude-haiku')
    parser.add_argument('--save_dir', type=str, default='/data/linjc/trialfm/downstream/summarization/results')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='cohort')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    

    with open(f'data/downstream/matching/patient2trial/{args.dataset}/test.json', 'r') as f:
        inputs = json.load(f)
    
    i = 0
    output_dict = {}
    
    # if os.path.exists(os.path.join(args.save_dir, f'{args.model_name}.json')):
    #     with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'r') as f:
    #         output_dict = json.load(f)

    for key, value in tqdm(inputs.items()):
        if key in output_dict:
            i += 1
            continue

        prompt = value['input']
        if args.model_name == 'claude-haiku':
            try:
                decoded = chat_haiku(prompt)
            except:
                decoded = ""
        elif args.model_name == 'claude-sonnet':
            try:
                decoded = chat_sonnet(prompt)
            except:
                decoded = ""
        else:
            raise ValueError(f"Model name {args.model_name} not supported")
        
        output_dict[key] = {
            'output': decoded,
            'label': value['label']
        }

        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)
    
        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)