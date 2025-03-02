import re
import random
import ast
import operator
import os
import sys
import json
import time
from collections import deque
from threading import Lock




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
        

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed


search_json_schema = {
    'title': 'query_augmentation',
    "type": "object",
    "properties": {
        "augmentated_query": {
            "type": "string",
            # "maxLength": 1000
        }
    },
    "required": ["augmentated_query"]
}


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



## retrieval engine for ms marco
sys.path.append('.')
import faiss
from index import search_faiss_index

# load ms marco passages
passages = []
with open("data/MS-MARCO/collection.tsv", "r") as f:
    for i, line in enumerate(f):
        p_id, passage = line.strip().split("\t")
        passages.append({"p_id": p_id, "passage": passage})
    
# load ms marco index
index = faiss.read_index("/projects/bdgw/langcao2/saved_indexes/ms_marco.index")


def run_retrieval_msmarco(query, topk=100):
    results = search_faiss_index(query, index, passages, topk)
    pid_list = [result[0]['p_id'] for result in results]

    return pid_list
    

def calculate_answer_score(answer_text, label, do_print=False):
    """Calculate answer score based on document recall."""
    try:
        data = json.loads(answer_text)
        query = data["augmentated_query"]
        
        retrieved_pids = run_retrieval_msmarco(query, topk=100)
            
        hit_pids = set(retrieved_pids) & set(label)
        recall = len(hit_pids) / len(label)
        
        if do_print:
            print(f"Recall: {recall}")
        
        if recall == 1:
            answer_score = 1
        elif 1 > recall > 0:
            answer_score = 0.5
        elif recall == 0:
            answer_score = 2
    
    except:
        print("[Error] Error in evaluation")
        answer_score = -2
    
    return answer_score


def compute_score(solution_str, ground_truth, format_reward=1, answer_reward=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    # label is a list of groundtruth pmids
    label = ground_truth['target']
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    format_correct = response_format_correct

    format_score = 1 if format_correct else -2

    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Target: {str(label)} |")
    
    answer_score = 0
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, do_print)

    total_score = format_score + answer_score
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score
    