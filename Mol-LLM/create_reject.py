import os

import random
import re
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from datasets import load_from_disk, Dataset

import selfies as sf

from data_utils import CLASSIFICATION_BENCHMARKS, REGRESSION_BENCHMARKS, REACTION_BENCHMARKS, MOL2TEXT_BENCHMARKS
from reject_utils import (
    load_tokenizer,
    clean_smiles_text,
    clean_target_text,
    fingerprint_similarity,
    text_similarity,
)


task_threshold = {
    # Classification. output is not used
    'smol-property_prediction-bbbp': {
        'input': [0.8, 0.1],
        'output': -1, 
    },
    'smol-property_prediction-clintox': {
        'input': [0.8, 0.1],
        'output': -1,
    },
    'smol-property_prediction-hiv': {
        'input': [0.8, 0.1],
        'output': -1,
    },
    'smol-property_prediction-sider': {
        'input': [0.8, 0.1],
        'output': -1,
    },
    
    
    # Regression
    'qm9_homo': {
        'input': [0.4, 0.1],
        'output': ,
    },
    'smol-property_prediction-esol': {
        'input': [0.4, 0.1],
        'output': ,
    },
    'smol-property_prediction-lipo': {
        'input': [0.6, 0.1],
        'output': ,
    }
    
    # Reaction prediction
    'forward_reaction_prediction': {
        'input': [, ],
        'output': ,
    }
    # Molecule captioning
    'chebi-20-mol2text': {
        'input': [, ],
        'output': ,
    }
}



def structural_similarity_map(data, dataset, args):
    n_reject = args.n_reject
    
    in_threshold, th_step = task_threshold[args.task]['input']
    out_threshold = task_threshold[args.task]['output']
    
    in_smiles = clean_smiles_text(data['input_mol_string'])
    in_target = clean_target_text(data['target_text'], args.task)
    
    candi_indicies = list(range(len(dataset)))
    
    used_indicies = []
    for i in range(n_reject):
        try_count = 0
        while True:
            try_count += 1
            if try_count > args.decrease_count:
                in_threshold -= th_step
                try_count = 0
            
            candi_idx = random.choice(candi_indicies)
            if dataset[candi_idx]['input_mol_string'] == data['input_mol_string']:
                continue
            elif candi_idx in used_indicies:
                continue
            
            out_smiles = clean_smiles_text(dataset[candi_idx]['input_mol_string'])
            out_target = clean_target_text(in_target, args.task)
            
            sim_score = fingerprint_similarity(in_smiles, out_smiles, method='tanimoto')
            if not sim_score > in_threshold:  # Check similarity between input and candidate
                continue
            
            if args.task in CLASSIFICATION_BENCHMARKS:
                out_condition = in_target != out_target  # binary classification
            elif args.task in REGRESSION_BENCHMARKS:
                out_condition = abs(in_target - out_target) > out_threshold
            elif args.task in REACTION_BENCHMARKS:
                out_sim_score = fingerprint_similarity(in_target, out_target, method='tanimoto')
                out_condition = abs(sim_score - out_sim_score) > out_threshold
            elif args.task in MOL2TEXT_BENCHMARKS:
                out_sim_score = text_similarity(in_target, out_target, method='bleu')
                out_condition = out_sim_score < out_threshold  # simiarity between 2 descriptions
            else:
                raise NotImplementedError(f"{args.task} is not supported for reject molecule generation.")
            
            if out_condition:
                rejected_data = dataset[candi_idx]
                
                data[f"{i}-th_rejected_selfies"] = sf.encoder(clean_smiles_text(dataset[candi_idx]['input_mol_string']))
                data[f"{i}-th_rejected_idx"] = sf.encoder(str(candi_idx))
                
                data[f"{i}-th_rejected_x"] = rejected_data["x"]
                data[f"{i}-th_rejected_edge_index"] = rejected_data["edge_index"]
                data[f"{i}-th_rejected_edge_attr"] = rejected_data["edge_attr"]
                data[f"{i}-th_additional_rejected_x"] = data['additional_x']
                data[f"{i}-th_additional_rejected_edge_index"] = data['additional_edge_index']
                data[f"{i}-th_additional_rejected_edge_attr"] = data['additional_edge_attr']
                used_indicies.append(candi_idx)
                break
            
    return data


    

def main(args):
    print(f"Start creating reject molecule for {args.task}")

    dataset_path = os.path.join(args.dataset_path, args.split, args.task)
    
    result_path = os.path.join(args.result_path, args.split, args.reject_method, args.task)
    os.makedirs(result_path, exist_ok=True)

    dataset = load_from_disk(dataset_path)
    # print(dataset[0].keys())
    
    print(f"Loaded dataset with {len(dataset)} samples.")
    print(f"Start creating reject molecule by {args.reject_method}")


    if args.reject_method == 'structure':
        reject_method = structural_similarity_map
    else:
        raise NotImplementedError(f'{args.reject_method} is not supported.')
    
    wrapped_func = partial(
        reject_method,
        dataset=dataset,
        args=args
    )

    print("Starting multiprocessing map...")
    new_dataset = dataset.map(
        wrapped_func,
        num_proc=args.num_proc,
        desc="Generating rejected samples"
    )

    new_dataset.save_to_disk(result_path)
    print(f"Saved new dataset with rejected molecules to {result_path}")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default='/app2/d4ft-textmol/nonn', help="Path to the dataset")
    
    parser.add_argument('--dataset_path', type=str, default='/data/data/task_data', help="Path to the dataset")
    parser.add_argument("--result_path", type=str, default='/data/data/smilar_data/far', help="Path to save the result")
    
    parser.add_argument("--split", type=str, default='train', help="Dataset split")
    # parser.add_argument("--split", type=str, default='test', help="Dataset split")
    # parser.add_argument("--task", type=str, default='qm9_homo', help="Task name")
    
    parser.add_argument("--n_reject", type=int, default=12, help="Number of rejected molecules to find")
    parser.add_argument('--reject_method', type=str, default='structure', choices='structure')
    parser.add_argument('--decrease_count', type=int, default=100, 
                        help='Decrease the threshold whenever the try count reachs decrease_count.')
    
    parser.add_argument("--num_proc", type=int, default=32, help="Number of processes for multiprocessing")
    args = parser.parse_args()

    
    tasks = ['smol-property_prediction-clintox', 'smol-property_prediction-hiv', 'smol-property_prediction-clintox']
    
    for task in tasks:    
        args.task = task
        main(args)
