import json
import argparse
import glob
import os
from tqdm import tqdm
from collections import defaultdict
import pdb
from datasets import load_dataset
import sys
sys.path.append('./')

import random
import pandas as pd
from src.dataset.matching.utils import read_trec_qrels


PROMPT = (
    "Hello. You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the "
    "eligibility criteria of a clinical trial to determine the patient's eligibility. "
    "The factors that allow someone to participate in a clinical study are called eligibility criteria. They are based on "
    "characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other "
    "medical conditions."
    "\n\n"
    "The assessment of eligibility has a three-point scale: " 
    "0) Would not refer this patient for this clinical trial; "
    "1) Would consider referring this patient to this clinical trial upon further investigation; and "
    "2) Highly likely to refer this patient for this clinical trial. \n"
    "You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility. "
    "\n\n"
)

input_patient_note_prefix = (
    "Here is the patient note:\n"
)
input_clinical_trial_prefix = (
    "Here is the clinical trial: \n"
)


def format_input(df_notes, ctgov_dict, qrels, ourput_filename):
    inputs = {}
    for idx, sample in enumerate(tqdm((qrels[:]))):
        patient_id, nct_id, label = sample
        # print type of patient_id
        patient_note = df_notes[df_notes['Patient ID'] == int(patient_id)]['Description'].values[0]
        # patient_note_sentences = convert_patient_note_into_sentences(patient_note)
        patient_note_sentences = patient_note
        
        if nct_id not in ctgov_dict:
            continue

        try:
            criteria_curr = ctgov_dict[nct_id]['eligibility_criteria']
            criteria_curr = criteria_curr.replace('~', '\n')
        except:
            criteria_curr = ''
        try:
            title_curr = ctgov_dict[nct_id]['brief_title']
        except:
            title_curr = ''
        try:
            target_diseases_curr = "Target diseases: " + ctgov_dict[nct_id]['conditions']
        except:
            target_diseases_curr = ''
        try:
            interventions_curr = "Interventions: " + ctgov_dict[nct_id]['interventions']
        except:
            interventions_curr = ''
        try:
            summary_curr = "Summary: " + ctgov_dict[nct_id]['brief_summary']
        except:
            summary_curr = ''
        
        input_clinical_trial = (
            f"{input_clinical_trial_prefix}"
            f"```Title: {title_curr}\n"
            f"{target_diseases_curr}\n"
            f"{interventions_curr}\n"
            f"{summary_curr}\n"
            f"Eligibility criteria\n{criteria_curr}```\n"
        )
        
        input_patient_note = (
            f"{input_patient_note_prefix}"
            f"```{patient_note_sentences}```"
        )

        input = PROMPT + input_patient_note + input_clinical_trial
        inputs[idx] = {
            'patient_id': patient_id,
            'nct_id': nct_id,
            'input': input,
            'label': label
        }
    
    with open(ourput_filename, 'w') as f:
        json.dump(inputs, f, indent=4, sort_keys=True)

    return inputs

def dataset_to_dict(dataset, key_column):
    """
    Convert a Hugging Face dataset to a dictionary using a specified column as the key.

    Args:
        dataset (Dataset): The Hugging Face dataset.
        key_column (str): The column to use as the dictionary key.

    Returns:
        dict: A dictionary where keys are from the specified column and values are the corresponding rows.
    """
    if key_column not in dataset.column_names:
        raise ValueError(f"Column '{key_column}' not found in dataset.")
    
    return {row[key_column]: row for row in tqdm(dataset)}

def load_ctgov_dict():
    dataset = load_dataset('linjc16/ctgov', split='train', num_proc=64)

    ctgov_dict = dataset_to_dict(dataset, 'nct_id')

    return ctgov_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_notes_file', type=str, default='data/matching/cohort/patient_notes.csv')
    parser.add_argument('--qrels_file', type=str, default='data/matching/cohort/qrels-clinical_trials.txt')
    parser.add_argument('--ourput_filename', type=str, default='data/matching/cohort/test.json')
    args = parser.parse_args()

    df_notes = pd.read_csv(args.patient_notes_file)
    
    qrels = read_trec_qrels(args.qrels_file)
    
    ctgov_dict = load_ctgov_dict()

    inputs = format_input(df_notes, ctgov_dict, qrels, args.ourput_filename)  

    print(f"Number of samples: {len(inputs)}")