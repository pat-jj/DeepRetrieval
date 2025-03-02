# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse


from random import randint, seed, choice
from tqdm import tqdm

def gen_dataset(
    N,
    DIGIT,
    LESS_OR_EQUAL=True,
):
    """
    any score <0.4 is ok, since +- is easy
    the model have 
    """
    seed(1)
    # Generate N pairs of numbers and their results for different operations
    equations = []
    # operations = ['*', '+', '-', '*', '*']
    operations = ['*']
    for _ in tqdm(range(N)):
        # Helper function to generate a number with 50% chance of being N-digit or N/2-digit
        def get_random_num():
            r = randint(1,3)
            if r == 0:
                # 2 digits less than original
                max_num = 10**(DIGIT-2)
                return randint(0 if LESS_OR_EQUAL else max_num//10, max_num-1)
            elif r == 1:
                # 1 digit less than original
                max_num = 10**(DIGIT-1)
                return randint(0 if LESS_OR_EQUAL else max_num//10, max_num-1)
            else:
                # N-digit number
                max_num = 10**DIGIT
                return randint(0 if LESS_OR_EQUAL else max_num//10, max_num-1)
        # Generate two numbers independently
        num1 = get_random_num()
        num2 = get_random_num()
        # Randomly choose operation
        op = choice(operations)
        # Calculate result based on operation
        if op == '*':
            result = num1 * num2
        elif op == '+':
            result = num1 + num2
        else:  # op == '-'
            assert op == '-'
            # For subtraction, ensure num1 >= num2
            if num1 < num2:
                num1, num2 = num2, num1
            result = num1 - num2
        equations.append((num1, num2, result, op))
    return equations
    
    
def make_prefix(dp):
    num1 = dp['num1']
    num2 = dp['num2']
    op = dp['operation']
    prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> RESULT_NUMBER </answer>. \nUser: Give me the answer of the following equation: {num1} {op} {num2}.\nAssistant: Let me solve this step by step.\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/arithmetic-3_digit')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'yolo/arithmetic-3_digit'
    DIGIT = 3
    # N = 1000000
    N = 100000
    LESS_OR_EQUAL = True
    TRAIN_SIZE = 32768
    TEST_SIZE = 4096

    dataset = gen_dataset(N=N, DIGIT=DIGIT, LESS_OR_EQUAL=LESS_OR_EQUAL)
    dataset = list(set(dataset))
    assert len(dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = dataset[:TRAIN_SIZE]
    test_dataset = dataset[-TEST_SIZE:]


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = make_prefix(example)
            solution = example['result']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
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
    
    def to_dataset(dataset_list):
        dataset_dict = {
            "num1": [],
            "num2": [],
            "result": [],
            "operation": []
        }
        for dp in dataset_list:
            dataset_dict["num1"].append(dp[0])
            dataset_dict["num2"].append(dp[1])
            dataset_dict["result"].append(dp[2])
            dataset_dict["operation"].append(dp[3])
        return Dataset.from_dict(dataset_dict)

    train_dataset = to_dataset(train_dataset)
    test_dataset = to_dataset(test_dataset)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
