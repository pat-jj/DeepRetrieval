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
The Assistant is a clinical specialist tasked with (1) Designing criteria (no more than 6, each should be unique) based on the User's input PICO, and (2) Assessing research papers for inclusion in a systematic literature review based on the designed criteria.

PICO is defined as follows:
P: Patient, Problem or Population - Who or what is the research about?
I: Intervention - What is the main intervention or exposure being considered?
C: Comparison - What is the intervention being compared to?
O: Outcome - What are the relevant outcomes or effects being measured?

The Assistant needs to evaluate each criterion of a paper given by the User to determine its eligibility for inclusion in the review. Provide a list of rationale-eligibility pairs for each designed criterion.
Decision Types:
1. YES (+1): Meets the criteria.
2. PARTIAL (+0.5): Partially meets the criteria but not completely.
3. UNCERTAIN (+0): Uncertain if it meets the criteria.
4. NO (-1): Does not meet the criteria.

The Assistant should:
1. Design criteria in JSON format within <criteria> </criteria> tags.
2. Explain the reasoning process freely in <think> </think> tags.
3. Provide evaluations in JSON format within <answer> </answer> tags.

Note:
If the information within the provided paper content is insufficient to conclusively evaluate a criterion, you should opt for "UNCERTAIN" as your response. Avoid making assumptions or extrapolating beyond the provided data, as accurate and reliable responses are crucial, and fabricating information (hallucinations) could lead to serious errors in the systematic review.
If the information is not applicable N/A, you should opt for "UNCERTAIN".
Use "PARTIAL" when the paper meets some aspects of the criterion but not all; ensure that the partial fulfillment is based on the provided data and not on assumptions or incomplete information.
Each pair of tags (<criteria> </criteria> and <think> </think> and <answer> </answer>) MUST ONLY APPEAR ONCE in the response.

For example:
==================Example==================
User: The PICO and the paper are as follows:

PICO:
P: Adult patients with type 2 diabetes
I: Metformin monotherapy
C: Other oral antidiabetic medications
O: HbA1c reduction and glycemic control

Paper:
Title: Comparative Efficacy of Metformin versus Sulfonylureas in Type 2 Diabetes Management
Abstract: In this 12-week randomized controlled trial, we evaluated the efficacy of metformin (n=150) compared to sulfonylureas (n=150) in patients aged 30-65 years with type 2 diabetes. Patients received either metformin (1000mg daily) or glibenclamide (5mg daily). Primary outcomes included changes in fasting plasma glucose and HbA1c levels. Secondary outcomes included weight changes and adverse events. Results showed comparable reductions in fasting glucose levels between groups, but HbA1c data was incomplete due to technical issues with lab processing. Both treatments were generally well-tolerated, with metformin group showing fewer hypoglycemic events.

Assistant: Let me solve this step by step.
<criteria>
{
    "criterion 1": "Study includes adult patients with type 2 diabetes",
    "criterion 2": "Intervention involves metformin monotherapy",
    "criterion 3": "Reports both HbA1c and glucose outcomes",
    "criterion 4": "Minimum one year follow up period",
    "criterion 5": "Sample size at least 300 patients",
    "criterion 6": "Reports complete safety data"
}
</criteria>
<think>
Analyzing the PICO elements to design appropriate criteria...
The study includes adult patients with type 2 diabetes, and the intervention is metformin monotherapy. However, the study duration is only 12 weeks, which is below the required one-year follow-up. The sample size is 300 patients, but the HbA1c data is incomplete, and safety data is limited to hypoglycemic events.
</think>
<answer>
{
    "evaluations": [
        {
            "rationale": "Study explicitly includes adults aged 30-65 years with type 2 diabetes",
            "eligibility": "YES"
        },
        {
            "rationale": "Patients received metformin 1000mg daily as monotherapy treatment",
            "eligibility": "YES"
        },
        {
            "rationale": "Reports glucose outcomes but HbA1c data is incomplete",
            "eligibility": "PARTIAL"
        },
        {
            "rationale": "Study duration was only 12 weeks, far below one year requirement",
            "eligibility": "NO"
        },
        {
            "rationale": "Total patient number not clearly stated for final analysis",
            "eligibility": "UNCERTAIN"
        },
        {
            "rationale": "No comprehensive safety data reported beyond hypoglycemic events",
            "eligibility": "NO"
        }
    ]
}
</answer>
==================End of Example==================

"""


def make_prefix(dp, template_type):
    if template_type == 'base':
        input_str = INSTRUCTION
        input_str += """
User: The PICO and the paper are as follows:
"""

    input_str += dp['input'] + """
Assistant: Let me solve this step by step.
<criteria>
"""

    return input_str


def format_pico_and_paper(pico, title, abstract):
    pico_str = "PICO:\n"
    pico_str += f"P: {pico['P']}\n"
    pico_str += f"I: {pico['I']}\n"
    pico_str += f"C: {pico['C']}\n"
    pico_str += f"O: {pico['O']}\n"
    
    paper_str = "Paper:\n"
    paper_str += f"Title: {title}\n"
    paper_str += f"Abstract: {abstract}\n"
    
    return pico_str + "\n" + paper_str



def load_matching_dataset():
    
    data_train = []

    with open('/home/pj20/server-04/LMR/Panacea-R1/data/pubmed_screen_origin/train.jsonl', 'r') as f:
        for line in f:
            data_train.append(json.loads(line))

    
    train_data = [{'input': format_pico_and_paper(x['pico'], x['title'], x['abstract']), 'label': 1 if x['label'] == 1 else -1} for x in data_train]
    
    # shuffle the dataset
    random.shuffle(train_data)
    
    return train_data, None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/literature_screening')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base', choices=['base', 'qwen-instruct', 'gpt'])

    args = parser.parse_args()
    
    data_source = 'literature_screening'
    
    train_data, _ = load_matching_dataset()

    train_dataset = Dataset.from_list(train_data)


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
                "ability": "literature_screening",
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

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)
    
    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

        
    print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")

    local_dir = os.path.join(args.local_dir, args.template_type)
    hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None
    
    os.makedirs(local_dir, exist_ok=True)
    
    
    # make a random split for validation set
    split_datasets = train_dataset.train_test_split(test_size=0.0005)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 