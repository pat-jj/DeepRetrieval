from tqdm import tqdm
import json
import argparse
import pandas as pd
import os
import sys
sys.path.append('./')
from src.dataset.matching.utils import read_trec_qrels
from src.dataset.matching.cohort.data_gen import load_ctgov_dict

import pdb


PROMPT = (
    "Hello. You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the "
    "eligibility criteria of a clinical trial to determine the patient's eligibility. "
    "The factors that allow someone to participate in a clinical study are called eligibility criteria. They are based on "
    "characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other "
    "medical conditions."
    "\n\n"
    "The assessment of eligibility has a three-point scale: " 
    "0) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria); "
    "1) Not relevant (patient does not have sufficient information to qualify for the trial); and "
    "2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply). \n"
    "You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility. "
    "\n\n"
)

input_patient_note_prefix = (
    "Here is the patient note:\n"
)
input_clinical_trial_prefix = (
    "Here is the clinical trial: \n"
)


def format_input(df_notes, ctgov_dict, qrels, output_filename):
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
    
    with open(output_filename, 'w') as f:
        json.dump(inputs, f, indent=4, sort_keys=True)

    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_notes_file', type=str, default='data/matching/TREC2021/patient_notes.csv')
    parser.add_argument('--qrels_file', type=str, default='data/matching/TREC2021/qrels-clinical_trials.txt')
    parser.add_argument('--output_filename', type=str, default='data/matching/TREC2021/test.json')
    args = parser.parse_args()
    
    df_notes = pd.read_csv(args.patient_notes_file)
    
    qrels = read_trec_qrels(args.qrels_file)
    
    ctgov_dict = load_ctgov_dict()

    inputs = format_input(df_notes, ctgov_dict, qrels, args.output_filename)

    print(f"Number of samples: {len(inputs)}")