import faiss
from tqdm import tqdm
import os
import sys
import json

sys.path.append('.')
from index import search_faiss_index, move_index_to_gpu



index_path = '/projects/bdgw/langcao2/saved_indexes/ms_marco.index'
    
# load index
index = faiss.read_index(index_path)

if faiss.get_num_gpus() > 0:
    index = move_index_to_gpu(index)

topk = 1000



if os.path.exists('outputs/baseline.json'):
    results = json.load(open('outputs/baseline.json', 'r'))
else:
    results = {}

queries = []
with open('data/MS-MARCO/queries.dev.tsv', 'r') as f:
    for line in f:
        qid, query = line.split('\t')
        queries.append((qid, query))


passages = []
with open('data/MS-MARCO/collection.tsv', 'r') as f:
    for line in f:
        pid, passage = line.split('\t')
        passages.append((pid, passage))


for i, (qid, query) in enumerate(tqdm(queries)):
    if qid in results:
        continue
    tmp_results = search_faiss_index(query, index, passages, k=topk)

    # results: [(p_id, passage, distance), ...]
    tmp_results = [[result[0][0], float(result[1])] for result in tmp_results]
    results[qid] = tmp_results

    # if i % 1000 == 0:
        # json.dump(results, open('outputs/baseline.json', 'w'))


json.dump(results, open('outputs/baseline.json', 'w'))

