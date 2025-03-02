import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb



INSTRUCTION = """
The Assistant is a clinical specialist. He is conducting research and doing a medical literature review. His task is to create query terms for a search URL to find relevant literature on PubMed or ClinicalTrials.gov.

The research is defined using the PICO framework:
P: Patient, Problem or Population - Who or what is the research about?
I: Intervention - What is the main intervention or exposure being considered?
C: Comparison - What is the intervention being compared to?
O: Outcome - What are the relevant outcomes or effects being measured?

"""


def make_prefix(dp, template_type):
    # input_str = dp['input']
    if template_type == 'base':
        input_str = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.\n""" + INSTRUCTION
        input_str += """The Assistant should show his thinking process in <think> </think> tags. The Assistant should return the final answer in JSON format in <answer> </answer> tags, 
For example:
<think>
[thinking process]
</think>
<answer>
{
    "query": "...."
} 
</answer>. 
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.

User: The research is defined by the following PICO:
"""

    input_str +=  dp['input'] + """
Assistant: Let me solve this step by step. 
<think>
"""

    # elif template_type == 'gpt':
    #     input_str += "Solve this step by step and return the final answer in <answer> </answer> tags, for example, <answer>  </answer>."
    # else:
    #     raise NotImplementedError

    return input_str


def convert_dict_to_str(pico_dict):
    pico_str = ""
    pico_str += f"P: {pico_dict['P']}\n"
    pico_str += f"I: {pico_dict['I']}\n"
    pico_str += f"C: {pico_dict['C']}\n"
    pico_str += f"O: {pico_dict['O']}\n"
    return pico_str



def load_matching_dataset():
    
    data_train = []
    data_test = []
    with open('/home/pj20/server-04/LMR/Panacea-R1/data/pubmed_search_origin/train.jsonl', 'r') as f:
        for line in f:
            data_train.append(json.loads(line))

    with open('/home/pj20/server-04/LMR/Panacea-R1/data/pubmed_search_origin/test.jsonl', 'r') as f:
        cnt = 0
        for line in f:
            data_test.append(json.loads(line))
            cnt += 1
            if cnt > 100:
                break
    
    train_data = [{'input': convert_dict_to_str(x['pico']), 'label': x['publication_pmids'], 'pub_date': x['pub_date']} for x in data_train]
    test_data = [{'input': convert_dict_to_str(x['pico']), 'label': x['publication_pmids'], 'pub_date': x['pub_date']} for x in data_test]
    
    return train_data, test_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/literature_mining')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base', choices=['base', 'qwen-instruct', 'gpt'])

    args = parser.parse_args()
    
    data_source = 'literature'
    
    train_data, test_data = load_matching_dataset()

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)


    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['label'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "literature_mining",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))
        
    print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")
    print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test)}")

    local_dir = os.path.join(args.local_dir, args.template_type)
    hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None
    
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 