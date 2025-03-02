import json



prediction_path = 'outputs/baseline.json'
predictions = json.load(open(prediction_path, 'r'))


ground_truth = {}

with open('data/MS-MARCO/qrels.dev.tsv', 'r') as f:
    for line in f:
        qid, group_id, pid, relevance = line.strip().split("\t")
        if relevance != '1':
            continue
        if qid in ground_truth:
            ground_truth[qid].append(pid)
        else:
            ground_truth[qid] = [pid]


recall_at_10 = 0
recall_at_100 = 0
recall_at_1000 = 0
total_queries = 0

# Calculate recall at 100 and 1000
for qid, prediction in predictions.items():
    # Get the retrieved pids from predictions
    retrieved_pids = [p_id for p_id, score in prediction]

    if qid in ground_truth:
        relevant_pids = set(ground_truth[qid])  # Ground truth relevant documents
        
        # Recall at 10
        relevant_in_top_10 = sum(1 for pid in retrieved_pids[:10] if pid in relevant_pids)
        recall_at_10 += relevant_in_top_10 / len(relevant_pids) if relevant_pids else 0
        
        # Recall at 100
        relevant_in_top_100 = sum(1 for pid in retrieved_pids[:100] if pid in relevant_pids)
        recall_at_100 += relevant_in_top_100 / len(relevant_pids) if relevant_pids else 0
        
        # Recall at 1000
        relevant_in_top_1000 = sum(1 for pid in retrieved_pids[:1000] if pid in relevant_pids)
        recall_at_1000 += relevant_in_top_1000 / len(relevant_pids) if relevant_pids else 0

        total_queries += 1

# Calculate average recall@100 and recall@1000
recall_at_10 = recall_at_10 / total_queries if total_queries > 0 else 0
recall_at_100 = recall_at_100 / total_queries if total_queries > 0 else 0
recall_at_1000 = recall_at_1000 / total_queries if total_queries > 0 else 0

# Print recall results
print(f'Prediction: {prediction_path}')
print(f'Total queries: {total_queries}')
print(f'Recall@10: {recall_at_10:.4f}')
print(f'Recall@100: {recall_at_100:.4f}')
print(f'Recall@1000: {recall_at_1000:.4f}')