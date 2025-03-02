import sys
import os

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Remove one dirname call to point to Panacea-R1
# Add the project root to Python path
sys.path.insert(0, project_root)  # This will now add Panacea-R1 to the path

print("Python path after:", sys.path)
print("Checking if verl exists:", os.path.exists(os.path.join(project_root, 'verl')))
print("Checking if utils exists:", os.path.exists(os.path.join(project_root, 'verl', 'utils')))
print("Checking if apis exists:", os.path.exists(os.path.join(project_root, 'verl', 'utils', 'apis')))

import json
# Import directly from the files instead of using the package structure
from verl.utils.apis.pubmed import PubmedAPI
from verl.utils.apis.ctgov import CTGovAPI
import re
from datetime import datetime
from tqdm import tqdm




def convert_date(date_str):
    # Default date if the input is empty or invalid
    default_date = "2025/02/12" # today

    if not date_str.strip():
        return default_date

    # Match patterns
    match = re.match(r'(\d{4})(?: (\w{3}))?(?: (\d{1,2}))?', date_str)
    if not match:
        return default_date

    year = match.group(1)
    month = match.group(2) if match.group(2) else '01'
    day = match.group(3) if match.group(3) else '01'

    # Convert month name to number
    if not month.isdigit():
        try:
            month = datetime.strptime(month, '%b').strftime('%m')
        except ValueError:
            return default_date

    # Format the date string to YYYY/MM/DD
    return f"{year}/{month.zfill(2)}/{day.zfill(2)}"


        

def get_pub_date(data):
    for item in tqdm(data):
        pmid = item['origin']['pmid']
        pub_date = pubmed_api.get_papers_by_pmids([pmid])['Publication Date'].fillna('').tolist()[0]
        item['pub_date'] = convert_date(pub_date)
    return data


if __name__ == '__main__':
    
    if os.path.exists('verl/utils/reward_score/apis/pubmed_api.key'):
        api_key = open('verl/utils/reward_score/apis/pubmed_api.key', 'r').read().strip()
        pubmed_api = PubmedAPI(api_key=api_key)
    ctgov_api = CTGovAPI()

    data_train = []
    data_test = []

    with open('/shared/rsaas/LEADS_project/LEADS_dataset/data/search/publication/train.jsonl', 'r') as f:
        for line in f:
            data_train.append(json.loads(line))

    with open('/shared/rsaas/LEADS_project/LEADS_dataset/data/search/publication/test.jsonl', 'r') as f:
        for line in f:
            data_test.append(json.loads(line))
            
    data_train = get_pub_date(data_train)
    data_test = get_pub_date(data_test)
    
    with open('/home/pj20/server-04/LMR/Panacea-R1/data/pubmed_search_origin/train.jsonl', 'w') as f:
        for item in data_train:
            f.write(json.dumps(item) + '\n')

    with open('/home/pj20/server-04/LMR/Panacea-R1/data/pubmed_search_origin/test.jsonl', 'w') as f:
        for item in data_test:
            f.write(json.dumps(item) + '\n')
    
    

