import faiss
import json
import torch
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Union, List
from tqdm import tqdm


tokenizer = None
model = None


sys.path.append("encoders/contriever")
from encoders.contriever.src.contriever import Contriever
model = Contriever.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

model = model.cuda()


def encode_text(text: Union[str, List[str]]):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        inputs = inputs.to(model.device)
        embeddings = model(**inputs)
    
    return embeddings.cpu().numpy()


def create_faiss_index(texts, batch_size=256):
    vectors = []
    first_batch = True
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_vectors = encode_text(batch_texts)

        # Convert to numpy array and add to list
        batch_vectors = batch_vectors

        # If it's the first batch, create the FAISS index
        if first_batch:
            dim = batch_vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            first_batch = False

        index.add(batch_vectors)

    return index, vectors


def move_index_to_gpu(index):
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    return index_gpu


def search_faiss_index(query, index, texts, k=3):

    query_vector = encode_text(query)
    distances, indices = index.search(query_vector, k)

    results = []
    for i in range(k):
        text = texts[indices[0][i]]
        distance = distances[0][i]
        results.append((text, distance))

    return results


def batch_search_faiss_index(queries, index, texts, k=3):

    query_vectors = encode_text(queries)
    distances, indices = index.search(query_vectors, k)

    results = []
    for i in range(len(queries)):
        query_results = []
        for j in range(k):
            text = texts[indices[i][j]]
            distance = distances[i][j]
            query_results.append((text, distance))
        results.append(query_results)

    return results



if __name__ == "__main__":

    # load data
    texts = []
    print('load data...')
    with open("data/MS-MARCO/collection.tsv", "r") as f:
        for i, line in enumerate(f):
            # if i > 100:
            #     break
            p_id, passage = line.strip().split("\t")
            texts.append({"p_id": p_id, "passage": passage})
    print('load done')

    # test data
    # texts = [
    #     {"p_id": "1", "passage": "This is the first passage."},
    #     {"p_id": "2", "passage": "This is the second passage."},
    #     {"p_id": "3", "passage": "Here is the third passage."},
    #     {"p_id": "4", "passage": "Another passage for testing."},
    # ]

    # index path
    # index_path = "saved_indexes/ms_marco.index" # too large for user space
    index_path = '/projects/bdgw/langcao2/saved_indexes/ms_marco.index'

    # create and save index
    # index, vectors = create_faiss_index([p['passage'] for p in texts])
    # faiss.write_index(index, index_path)
    
    # load index
    index = faiss.read_index(index_path)
    index = move_index_to_gpu(index)

    ##### retrieve
    query = "Find the second passage."
    print('start retrieve')
    results = search_faiss_index(query, index, texts, k=3)
    print('end retrieve')

    print(f"Query: {query}")
    print("\nTop 3 most similar passages:")
    for result in results:
        print(f"Passage ID: {result[0]['p_id']}, Passage: {result[0]['passage']}, Similarity: {result[1]:.4f}")
