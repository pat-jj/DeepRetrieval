import json
import sys
import os
from tqdm import tqdm
from cmqr_inference import CMQR
sys.path.append('./')

from src.eval.baselines.cmqr.cmqr_inference import CMQR
from src.Lucene.msmarco_beir.search import PyseriniMultiFieldSearch
from src.Lucene.utils import ndcg_at_k
from src.eval.BM25.utils import parse_qrel

# Initialize CMQR model
cmqr = CMQR()

# Check if the Pyserini index exists
if not os.path.exists("data/local_index_search/msmarco_beir/pyserini_index"):
    print("[Warning] Pyserini index not found for msmarco_beir")
    search_system = None
else:
    search_system = PyseriniMultiFieldSearch(index_dir="data/local_index_search/msmarco_beir/pyserini_index")

if __name__ == '__main__':
    # Load qrels file
    with open("data/raw_data/msmarco_beir/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = [line.strip().split("\t") for line in file]
    qrel_test = qrel_test[1:]  # Skip header
    qrel_test = parse_qrel(qrel_test)

    # Load queries from JSONL
    with open("data/raw_data/msmarco_beir/queries.jsonl", "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]
    queries_dict = {q['_id']: q['text'] for q in queries}

    # Build test dataset
    test_data = []
    for qid, value in qrel_test.items():
        test_data.append({
            "qid": qid,
            "query": queries_dict[qid],
            "target": value['targets'],
            "score": value['scores']
        })

    ndcg = []
    batch_size = 100

    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]

        rewritten_queries = []
        targets = {}
        scores = {}

        # Rewrite each query using CMQR
        for item in batch:
            rewritten = cmqr.rewrite(item['query'])
            rewritten_queries.append(rewritten)
            targets[rewritten] = item['target']
            scores[rewritten] = item['score']

        # Perform search using rewritten queries
        results = search_system.batch_search(rewritten_queries, top_k=10, threads=16)

        # Compute NDCG@10 for each rewritten query
        for query in rewritten_queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10, rel_scores=scores[query]))

    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")
