import ir_datasets
import pandas as pd
import os
import pdb


if __name__ == '__main__':

    dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")


    # patient notes
    patient_notes = []
    for query in dataset.queries_iter():
        # namedtuple<query_id, text>

        patient_notes.append((query.query_id, query.text.strip()))
    
    # save to csv in data/downstream/matching/patient2trial/TREC2021/
    df_patient_notes = pd.DataFrame(patient_notes, columns=['Patient ID', 'Description'])

    df_patient_notes.to_csv('data/downstream/matching/patient2trial/TREC2021/patient_notes.csv', index=False)
    
    
    qrel_save_dir = 'data/downstream/matching/patient2trial/TREC2021/'
    qrel_test_path = os.path.join(qrel_save_dir, 'qrels-clinical_trials.txt')
    qrel_train_path = os.path.join(qrel_save_dir, 'qrels-clinical_trials_train.txt')

    # split df_patient_notes query_id into train and test
    queries = df_patient_notes['Patient ID'].unique()

    # split 80% train, 20% test
    train_queries = queries[:int(len(queries)*0.8)]
    test_queries = queries[int(len(queries)*0.8):]

    # qrels-clinical_trials
    for rel in dataset.qrels_iter():
        # query_id, doc_id, relevance, iteration
        # save to csv in data/downstream/matching/patient2trial/TREC2021/qrels-clinical_trials.txt, split by \t
        if rel.query_id in train_queries:
            with open(qrel_train_path, 'a') as f:
                f.write(f"{rel.query_id}\t{rel.iteration}\t{rel.doc_id}\t{rel.relevance}\n")
        else:
            with open(qrel_test_path, 'a') as f:
                f.write(f"{rel.query_id}\t{rel.iteration}\t{rel.doc_id}\t{rel.relevance}\n")

        