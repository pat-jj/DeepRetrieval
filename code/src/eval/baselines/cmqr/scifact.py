import json
import sys
import os
from tqdm import tqdm
sys.path.append('./')
from src.eval.baselines.cmqr.cmqr_inference import CMQR

from src.Lucene.scifact.search import PyseriniMultiFieldSearch
from src.Lucene.utils import ndcg_at_k

# Initialize the CMQR model
cmqr = CMQR()

# Check if the Pyserini index exists
if not os.path.exists("data/local_index_search/scifact/pyserini_index"):
    print("[Warning] Pyserini index not found for scifact")
    search_system = None
else:
    search_system = PyseriniMultiFieldSearch(index_dir="data/local_index_search/scifact/pyserini_index")

if __name__ == '__main__':
    # Load qrels from the test set
    with open("data/raw_data/scifact/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = [line.strip().split("\t") for line in file]
    qrel_test = qrel_test[1:]  # Skip header line

    # Load queries from JSONL file
    with open("data/raw_data/scifact/queries.jsonl", "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]
    queries_dict = {q['_id']: q['text'] for q in queries}

    # Build the test dataset
    test_data = []
    for qid, docid, label in qrel_test:
        test_data.append({
            "qid": qid,
            "query": queries_dict[qid],
            "target": docid,
            "label": int(label)
        })

    ndcg = []
    batch_size = 100

    # Process test data in batches
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        
        # Rewrite queries using CMQR
        rewritten_queries = []
        targets = {}
        for item in batch:
            rewritten = cmqr.rewrite(item['query'])
            rewritten_queries.append(rewritten)
            targets[rewritten] = item['target']  # Map rewritten query to its target doc ID
        
        # Perform search using rewritten queries
        results = search_system.batch_search(rewritten_queries, top_k=10, threads=16)
        
        # Compute NDCG@10 for each query
        for query in rewritten_queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10))

    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")
