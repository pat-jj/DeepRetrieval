import json
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='cohort')
    args = parser.parse_args()
    with open(f'data/downstream/matching/patient2trial/{args.dataset}/{args.split}_cot.json', 'r') as f:
        inputs = json.load(f)
    
    
    inputs_new = {}
    for key, value in inputs.items():
        # change input "Let's think step by step. \nFinally, you should always repeat" -> "Finally, you only need to output"
        value['input'] = value['input'].replace("Let's think step by step. \nFinally, you should always repeat", "Don't output any explanations. You need to output only")
        inputs_new[key] = value
    
    with open(f'data/downstream/matching/patient2trial/{args.dataset}/{args.split}.json', 'w') as f:
        json.dump(inputs_new, f, indent=4, sort_keys=True)