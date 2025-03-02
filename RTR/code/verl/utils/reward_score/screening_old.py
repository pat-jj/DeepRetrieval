import re
import random
import json


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
        

def calculate_answer_score(answer_text, label, do_print=False):
    """Calculate answer score based on document recall."""
    try:
        # Parse the answer JSON
        answer_data = json.loads(answer_text)
        pred_evaluation = answer_data["evaluations"]
        
        # Calculate average evaluation score
        total_score = 0
        for eval_item in pred_evaluation:
            eligibility = eval_item["eligibility"]
            if eligibility == "YES":
                total_score += 1
            elif eligibility == "PARTIAL":
                total_score += 0.5
            elif eligibility == "UNCERTAIN":
                total_score += 0
            elif eligibility == "NO":
                total_score += -1
                
        avg_score = total_score / len(pred_evaluation)
        
        # Multiply by label to reward/punish appropriately
        answer_score = avg_score * label * 5
        
        if do_print:
            print(f"Average evaluation score: {avg_score}")
            print(f"Final answer score after label multiplication: {answer_score}")
            
        return answer_score

    except Exception as e:
        if do_print:
            print(f"[Error] Error in evaluation: {str(e)}")
        return -4

def validate_criteria_design(processed_str, do_print=False):
    """Validates the criteria design in the thinking section.
    
    Returns:
        Tuple of (validation_passed, list of error messages)
    """
    errors = []
    try:
        # Extract content between think tags
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, processed_str, re.DOTALL)
        if not think_match:
            if do_print:
                print("[Error] No thinking section found")
            errors.append("missing_think_content")
            return False, errors
            
        think_content = think_match.group(1)
        
        try:
            # Find JSON in thinking section
            criteria_json = extract_json_from_llm_output(think_content)
        except:
            if do_print:
                print("[Error] Failed to parse JSON in thinking section")
            errors.append("invalid_json")
            return False, errors
        
        # Validate criteria
        if not isinstance(criteria_json, dict):
            if do_print:
                print("[Error] Criteria not in JSON format")
            errors.append("not_json_dict")
            return False, errors
            
        criteria = list(criteria_json.values())
        
        # Check number of criteria
        if len(criteria) > 7:
            if do_print:
                print("[Error] Too many criteria (>7)")
            errors.append("too_many_criteria")
            return False, errors
            
        # Check uniqueness
        if len(set(criteria)) != len(criteria):
            if do_print:
                print("[Error] Criteria not unique")
            errors.append("duplicate_criteria")
            return False, errors
            
        # Check token limits (5-20 tokens per criterion)
        for criterion in criteria:
            token_count = len(criterion.split())
            if token_count < 2 or token_count > 30:
                if do_print:
                    print(f"[Error] Criterion token count ({token_count}) outside limits: {criterion}")
                errors.append("token_limit")
                return False, errors
                
        return True, []
        
    except Exception as e:
        if do_print:
            print(f"[Error] Error validating criteria: {str(e)}")
        errors.append("general_error")
        return False, errors

def compute_format_score(structure_errors, criteria_errors, do_print=False):
    """Compute format score based on different types of violations."""
    penalties = {
        # Structure penalties
        # "tag_count_think_start": -2,
        # "tag_count_think_end": -2,
        # "tag_count_answer_start": -2,
        # "tag_count_answer_end": -2,
        # "tag_order": -2,
        
        # Criteria penalties
        "missing_think_content": -3,
        "invalid_json": -3,
        "not_json_dict": -3,
        "too_many_criteria": -2,
        "duplicate_criteria": -2,
        "token_limit": -2,
        "general_error": -4
    }
    
    total_penalty = 0
    for error in structure_errors + criteria_errors:
        penalty = penalties.get(error, -1)
        total_penalty += penalty
        if do_print:
            print(f"Applied penalty {penalty} for error: {error}")
    
    # Cap the minimum score at -4
    return max(-4, total_penalty) if total_penalty != 0 else 1

def compute_score(solution_str, ground_truth):
    """The scoring function for screening task."""
    label = ground_truth['target']
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1

    # Validate both response structure and criteria design
    response_format_correct = validate_response_structure(processed_str, do_print)
    criteria_format_correct, criteria_errors = validate_criteria_design(processed_str, do_print)
    format_correct = response_format_correct and criteria_format_correct

    # format_score = compute_format_score(structure_errors, criteria_errors, do_print)
    if format_correct:
        format_score = 1
    else:
        if not response_format_correct:
            format_score = -8
        elif response_format_correct and not criteria_format_correct:
            format_score = -5
    
    
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
    