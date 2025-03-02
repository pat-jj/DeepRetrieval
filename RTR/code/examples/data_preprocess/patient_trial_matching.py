"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

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


def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        
        samples.append((target, numbers))
    
    return samples

# def make_prefix_old(dp, template_type):
#     target = dp['target']
#     numbers = dp['nums']
#     # NOTE: also need to change reward_score/countdown.py
#     if template_type == 'base':
#         """This works for any base model"""
#         prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
# User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
# Assistant: Let me solve this step by step.
# <think>"""
#     elif template_type == 'qwen-instruct':
#         """This works for Qwen Instruct Models"""
#         prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
#     return prefix

# def make_prefix(dp, template_type):
#     input_str = dp['input']
#     if template_type == 'base':
#         input_str = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n""" + input_str
#         input_str += """\nShow your work in <think> </think> tags. Before making a decision, you should analyze the eligibility of the given patient in a criterion-by-criterion style. For each criterion, you should provide: (1) the explanation of the patient-criterion relevance; (2) the locations of relevant sentences in the patient notes to the criterion, (3) the eligibility classification for each patient-criterion pair, and (4) final decision by aggregating the criterion-level results. You should return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible. </answer>. 
# Assistant: Let me solve this step by step. 
# <think>"""
#     elif template_type == 'qwen-instruct':
#         input_str = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
#         input_str += """\nShow your work in <think> </think> tags. Before making a decision, you should analyze the eligibility of the given patient in a criterion-by-criterion style. For each criterion, you should provide: (1) the explanation of the patient-criterion relevance; (2) the locations of relevant sentences in the patient notes to the criterion, (3) the eligibility classification for each patient-criterion pair, and (4) final decision by aggregating the criterion-level results. You should return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible. </answer>.<|im_end|>
# <|im_start|>assistant\nLet me solve this step by step.\n<think>"""
#     else:
#         raise NotImplementedError

#     return input_str


# def make_prefix(dp, template_type):
#     input_str = dp['input']
#     if template_type == 'base':
#         input_str = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n""" + input_str
#         input_str += """\nShow your work in <think> </think> tags. You should return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible. </answer>. 
# Assistant: Let me solve this step by step. 
# <think>"""
#     elif template_type == 'qwen-instruct':
#         input_str = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
#         input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
# <answer>
# {
#     "inclusion_analysis": [
#         {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "included"}, # choose from "included", "not included", "not enough information", "not applicable"
#         {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "xxx"}
#     ],
#     "exclusion_analysis": [
#         {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "not excluded"}, # choose from "excluded", "not excluded", "not enough information", "not applicable"
#         {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "xxx"}
#     ],
#     "final_prediction": {
#         "idx": 2,
#         "prediction": "Eligible"
#     }
# }
# </answer>.<|im_end|>
# <|im_start|>assistant\nLet me solve this step by step.\n<think>"""
#     elif template_type == 'gpt':
#         input_str += "Solve this step by step and return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible.  </answer>."
#     else:
#         raise NotImplementedError

#     return input_str


def make_prefix(dp, template_type):
    input_str = dp['input']
    if template_type == 'base':
        input_str = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n""" + input_str
        input_str += """\nShow your work in <think> </think> tags. You should return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible. </answer>. 
Assistant: Let me solve this step by step. 
<think>"""
    elif template_type == 'qwen-instruct':
        input_str = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
        input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{
    "inclusion_analysis": [
        {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "included"}, # choose from "included", "not included", "not enough information", "not applicable"
        {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "xxx"}
    ],
    "exclusion_analysis": [
        {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "not excluded"}, # choose from "excluded", "not excluded", "not enough information", "not applicable"
        {"criterion": "xxx", "analysis": "xxx", "eligibility_prediction": "xxx"}
    ],
    "reasoning": {
        "Step 1": "xxx",
        "Step 2": "xxx",
        "Step x": "xxx"
    }
    "final_prediction": {
        "idx": 2,
        "prediction": "Eligible"
    }
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    elif template_type == 'gpt':
        input_str += "Solve this step by step and return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible.  </answer>."
    else:
        raise NotImplementedError

    return input_str

def load_matching_dataset():
    # load data/matching/TREC2021/train.json
    with open('data/matching/TREC2021/train.json', 'r') as f:
        train_data_dict = json.load(f)

    with open('data/matching/cohort/test.json', 'r') as f:
        sigir_dict = json.load(f)
    
    with open('data/matching/TREC2021/test.json', 'r') as f:
        trec_dict = json.load(f)

    train_data = [{'input': x['input'], 'label': x['label']} for _, x in train_data_dict.items()]
    test_data = [{'input': x['input'], 'label': x['label']} for _, x in sigir_dict.items()]
    test_data.extend([{'input': x['input'], 'label': x['label']} for _, x in trec_dict.items()])
    
    return train_data, test_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/matching')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct', choices=['base', 'qwen-instruct', 'gpt'])

    args = parser.parse_args()
    
    data_source = 'matching'
    
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
                "ability": "patient_trial_matching",
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

    # only select random 1000 samples for test
    # test_dataset = test_dataset.select(range(5000))

    def balance_class(data):
        counter_labels = Counter([x['label'] for x in data])
        print(f'Original label distribution: {counter_labels}')
        min_samples = min(counter_labels.values())

        label_counts = {label: 0 for label in counter_labels.keys()}
        def balance_class(example, indices, label_counts):
            label = example['label']
            if label_counts[label] < min_samples:
                label_counts[label] += 1
                return True
            return False
        
        data = data.filter(
            lambda example, idx: balance_class(example, idx, label_counts),
            with_indices=True
        )

        new_counter_labels = Counter(data['label'])
        print(f'New label distribution: {new_counter_labels}')

        return data

    train_dataset = balance_class(train_dataset)
    test_dataset = balance_class(test_dataset)


    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))

    # # get the distribution of lengths
    # import matplotlib.pyplot as plt
    # plt.hist(lengths_list, bins=100)
    # plt.title('Lengths of training examples')
    # plt.savefig('train_lengths.png')
    # plt.close()

    # plt.hist(lengths_list_test, bins=100)
    # plt.title('Lengths of test examples')
    # plt.savefig('test_lengths.png')
    # plt.close()
    threshold = 1024
    
    def truncate(train_dataset):
        count = 0
        # for those that are exceeding the threshold, we can delete the text between "\nTitle:" to "\nInclusion criteria:"
        for i, d in enumerate(train_dataset):
            if len(d['prompt'][0]['content'].split()) > threshold:
                text = d['prompt'][0]['content']
                start = text.find("\n```Title:")
                end = text.find("\nEligibility criteria:")
                new_text = text[:start] + text[end:]
                train_dataset[i]['prompt'][0]['content'] = new_text

                count += 1

                if len(new_text.split()) > threshold:
                    words = new_text.split()
                    truncate_length = max(threshold - 200, 0)  # Ensure we don't end up with a negative index
                    new_text = ' '.join(words[-truncate_length:])
                    train_dataset[i]['prompt'][0]['content'] = new_text

        print(f"Truncated {count} examples")

        return train_dataset
    
    train_dataset = truncate(train_dataset)
    test_dataset = truncate(test_dataset)
    
    local_dir = os.path.join(args.local_dir, args.template_type)
    hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None

    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
