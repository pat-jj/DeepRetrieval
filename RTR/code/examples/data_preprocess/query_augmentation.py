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



INSTRUCTION = """The Assistant is a query augmenter. It enhances the query to enrich it, which will then be used to retrieve relevant passages for the query."""

ANSWER_FORMAT = """
{
"augmented_query": "...."
} 
"""

USER_PREFIX = """The original query is: """


def make_prefix(dp, template_type):
    # if template_type == 'base':
    input_str = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.
{INSTRUCTION}

The Assistant should show his thinking process in <think> </think> tags. The Assistant should return the final answer in JSON format in <answer> </answer> tags, 
For example:
<think>
[thinking process]
</think>
<answer>
{ANSWER_FORMAT}
</answer>. 

User: {USER_PREFIX + dp['input']}
Assistant: Let me solve this step by step. 
<think>
"""
    return input_str





def load_matching_dataset():

    train_queries = {}
    test_queries = {}

    train_qrels = {}
    test_qrels = {}

    with open('data/MS-MARCO/queries.train.tsv', 'r') as f:
        for line in f:
            qid, query = line.strip().split("\t")
            train_queries[qid] = query

    with open('data/MS-MARCO/queries.dev.tsv', 'r') as f:
        for line in f:
            qid, query = line.strip().split("\t")
            test_queries[qid] = query

    with open('data/MS-MARCO/qrels.train.tsv', 'r') as f:
        for line in f:
            qid, group_id, pid, relevance = line.strip().split("\t")
            if relevance != '1':
                continue
            if qid in train_qrels:
                train_qrels[qid].append(pid)
            else:
                train_qrels[qid] = [pid]

    with open('data/MS-MARCO/qrels.dev.tsv', 'r') as f:
        for line in f:
            qid, group_id, pid, relevance = line.strip().split("\t")
            if relevance != '1':
                continue
            if qid in test_qrels:
                test_qrels[qid].append(pid)
            else:
                test_qrels[qid] = [pid]


    train_data = []
    test_data = []

    for qid, query in train_queries.items():
        if qid in train_qrels:
            train_data.append({
                "qid": qid,
                "input": query,
                "label": train_qrels[qid]
            })

    for qid, query in test_queries.items():
        if qid in test_qrels:
            test_data.append({
                "qid": qid,
                "input": query,
                "label": test_qrels[qid]
            })

    return train_data, test_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/MS-MARCO')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base', choices=['base', 'qwen-instruct', 'gpt'])

    args = parser.parse_args()
    
    data_source = 'ms-marco'
    
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
                "ability": "query_augmentation",
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