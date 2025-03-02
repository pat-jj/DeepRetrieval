import re
import random
import ast
import operator
import pdb
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

def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                print("[Error] Empty JSON string")
            return False
        
        data = json.loads(json_str)
        
        # Required keys
        # required_keys = {"inclusion_analysis", "exclusion_analysis", "final_prediction"}
        required_keys = {"final_prediction", 'reasoning', "inclusion_analysis", "exclusion_analysis"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        # Check inclusion_analysis list structure
        if not isinstance(data["inclusion_analysis"], list) or not all(
            isinstance(item, dict) and "criterion" in item and "analysis" in item and "eligibility_prediction" in item
            for item in data["inclusion_analysis"]
        ):
            print("[Error] Inclusion analysis structure is incorrect")
            return False
        
        # Check exclusion_analysis list structure (can be empty)
        if not isinstance(data["exclusion_analysis"], list) or not all(
            isinstance(item, dict) and "criterion" in item and "analysis" in item and "eligibility_prediction" in item
            for item in data["exclusion_analysis"]
        ):
            if len(data["exclusion_analysis"]) > 0:
                print("[Error] Exclusion analysis structure is incorrect")
                return False

        # Check final_prediction structure
        if not isinstance(data["final_prediction"], dict) or not all(
            key in data["final_prediction"] for key in ["idx", "prediction"]
        ):
            if do_print:
                print("[Error] Final prediction structure is incorrect")
            return False

        # Ensure idx is a number
        if not isinstance(data["final_prediction"]["idx"], int):
            if do_print:
                print("[Error] Final prediction 'idx' is not an integer")
            return False

        return True
    except json.JSONDecodeError:
        if do_print:
            print("[Error] JSON decoding failed")
        return False

# def cal_ans_score_old(format_correct, answer_text, label, do_print=False):
#     answer_score = 0
#     if format_correct and answer_text:
#         try:
#             pred_number_match = re.findall(r'\d+', answer_text)
#             # pred_text_match = re.search(r'(Excluded|Not relevant|Eligible)', answer_text, re.IGNORECASE)
            
#             # make sure the pred_number is only one, i.e., only 0, instead of 0, 1, 2
#             # if not, return 0
#             if len(pred_number_match) > 1:
#                 if do_print:
#                     print(f"More than one number found")
#                 answer_score = -2
#             else:
#                 pred_number = pred_number_match[0] if pred_number_match else None
#                 # pred_text = pred_text_match.group(1).capitalize() if pred_text_match else None
#                 if pred_number == label:
#                     answer_score = 2
#                     if do_print:
#                         print(f"Correct answer")
#                 else:
#                     if do_print:
#                         print(f"Wrong answer")
#                     answer_score = -1.5
#         except:
#             print(f"Error in evaluation")
#             answer_score = -2

#     return answer_score
    
def calculate_answer_score(json_str, label, do_print=False):
    """Calculate answer score based on final_prediction idx."""
    try:
        data = json.loads(json_str)
        pred_number = data["final_prediction"].get("idx")
        if pred_number == label:
            answer_score = 2
            if do_print:
                print("Correct answer")
        else:
            answer_score = -1.5
            if do_print:
                print("Wrong answer")
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

    label = str(ground_truth['target'])
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    # json_format_correct = check_json_format(answer_text, do_print)
    # format_correct = response_format_correct and json_format_correct
    format_correct = response_format_correct

    format_score = format_reward if format_correct else -abs(format_reward)
    # if do_print:
    #     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    #     print(f"Format score: {format_score}")
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Target: {label} |")
    
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
    

# def test_check_json_format():
#     valid_json = '{"inclusion_analysis": [{"criterion": "Age requirement", "analysis": "Meets age requirement", "eligibility_prediction": "included"}], "exclusion_analysis": [{"criterion": "Liver disease", "analysis": "Has liver disease", "eligibility_prediction": "excluded"}], "final_prediction": {"idx": 1, "prediction": "Ineligible"}}'
#     valid_json_empty_exclusion = '{"inclusion_analysis": [{"criterion": "Age requirement", "analysis": "Meets age requirement", "eligibility_prediction": "included"}], "exclusion_analysis": [], "final_prediction": {"idx": 2, "prediction": "Eligible"}}'
#     invalid_json_1 = '{"inclusion_analysis": "wrong format", "final_prediction": {"idx": 1, "prediction": "Eligible"}}'
#     invalid_json_2 = '{"inclusion_analysis": [{"criterion": "Age requirement", "analysis": "Meets age requirement"}], "exclusion_analysis": [], "final_prediction": {"idx": "one", "prediction": "Eligible"}}'
#     invalid_json_3 = 'Not a JSON string'
    
#     assert check_json_format(valid_json) == True, "Test case 1 failed"
#     assert check_json_format(valid_json_empty_exclusion) == True, "Test case 2 failed"
#     assert check_json_format(invalid_json_1) == False, "Test case 3 failed"
#     assert check_json_format(invalid_json_2) == False, "Test case 4 failed"
#     assert check_json_format(invalid_json_3) == False, "Test case 5 failed"

#     print("All test cases passed!")

# if __name__ == '__main__':
# #     sol_str = """1212 Assistant: Let me solve this step by step.
# # [36m(main_task pid=2123691)[0m <think>First, let's break down the clinical trial's inclusion criteria: The patient must be 18 years or older, have a healthy condition, and have an indication of asymptomatic bilateral extractions of lower third molars.
# # [36m(main_task pid=2123691)[0m Next, let's examine the patient note to see if they meet these criteria.
# # [36m(main_task pid=2123691)[0m The patient is a 60-year-old man with a history of frequent headaches, generalized bone pain, and difficulty chewing. Despite this, he still seems to have an indication of asymptomatic bilateral extractions of lower third molars. However, the note does not specifically mention that he has a good health condition, which is a critical inclusion criterion for the clinical trial.
# # [36m(main_task pid=2123691)[0m Based on these findings, it seems that the exclusion criteria of the clinical trial were likely satisfied, as the patient does not meet all the necessary criteria for participation.</think>
# # [36m(main_task pid=2123691)[0m <answer>Trial-level eligibility: 0.</answer><|endoftext|>
# # """
# #     gt = {'target': 0}
# #     print(compute_score(sol_str, gt))

#     test_check_json_format()