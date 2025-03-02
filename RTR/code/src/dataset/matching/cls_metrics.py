import json
import re
import os
import argparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='eval_results_qwen2.5-3B-instruct')
    parser.add_argument('--res_dir', type=str, default='results')
    args = parser.parse_args()

    args.res_dir = os.path.join(args.res_dir, f'{args.model_name}.json')

    with open(args.res_dir, 'r') as f:
        results = json.load(f)

    # change the ['']['output'] to ['']['generated_text']
    # change the ['']['label'] to ['']['target']
    # if exists
        
    for id, value in results.items():
        if 'output' in value:
            value['generated_text'] = value.pop('output')
        if 'label' in value:
            value['target'] = value.pop('label')
    
    # extract pred from value['output']
    preds = []
    labels = []
    for key, value in results.items():
        solution_str = value['generated_text']
        # find "assistant\nLet me solve this step by step.", split by this string, and get the second part
        if args.model_name != 'gpt-4o':
            solution_str = solution_str.split("assistant\nLet me solve this step by step.")[1]

        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, solution_str, re.DOTALL)  # Use re.DOTALL to match multiline content
        if matches:
            solution_extract = matches[-1].strip()
        else:
            solution_extract = ""

        pred_number_match = re.findall(r'\d+', solution_extract)

        pred = int(pred_number_match[0]) if pred_number_match else -1

        preds.append(pred)
        
        labels.append(value['target'])
    
    
    assert len(preds) == len(labels)

    # calculate accuracy using sklearn

    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy:.4f}")
    
    # calculate precision, recall, f1-score using sklearn and the results using two decimal places

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    
    data_temp = []
    for cls in [0, 1, 2]:
        if str(cls) in report:
            print(f"Recall for class {cls}: {report[str(cls)]['recall']:.4f}")
            print(f'Precision for class {cls}: {report[str(cls)]["precision"]:.4f}')
            print(f'F1-score for class {cls}: {report[str(cls)]["f1-score"]:.4f}')

            data_temp.append(f'{report[str(cls)]["f1-score"]:.4f}')
    
    # print(','.join(data_temp))
    # print weighted accuracy

    balanced_accuracy = balanced_accuracy_score(labels, preds)
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    
    # calculate kappa score
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(labels, preds)
    print(f"Kappa: {kappa:.4f}")

    # calculate Krippendorff's alpha-reliability