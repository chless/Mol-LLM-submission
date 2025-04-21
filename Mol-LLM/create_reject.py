import os
import argparse
import bisect
import random
import re
from itertools import chain
from functools import partial
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset

import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from data_utils import CLASSIFICATION_BENCHMARKS, REGRESSION_BENCHMARKS, REACTION_BENCHMARKS, MOL2TEXT_BENCHMARKS

task2range = {
    # Classification
    'bace': {
        
    },
    'smol-property_prediction-bbbp': {
        
    },
    'smol-property_prediction-clintox': {
        
    },
    'smol-property_prediction-hiv': {
        
    },
    'smol-property_prediction-sider': {
        
    },
    
    # Regression
    'qm9_homo': {
        'start_value': -0.43,
        'end_value': -0.11,
        'step_size': 0.02,
    },
    'smol-property_prediction-esol': {
        'start_value': -12,
        'end_value': 2,
        'step_size': 0.5,
    },
    'smol-property_prediction-lipo': {
        'start_value': -2,
        'end_value': 5,
        'step_size': 0.25,
    },
    
    # forward reaction prediction
    "forward_reaction_prediction":{
        
    }
    
    # Mol2Text
    
}

task_threshold = {
    # Classification
    'smol-property_prediction-bbbp': {
        'structure': [0.8, 0.05, 0.1],
    },
    'smol-property_prediction-clintox': {
        'structure': [0.8, 0.05, 0.1],
    },
    'smol-property_prediction-hiv': {
        'structure': [0.8, 0.05, 0.1],
    },
    'smol-property_prediction-sider': {
        'structure': [0.8, 0.05, 0.1],
    },
    
    
    # Regression
    'qm9_homo': {
        'structure': [0.4, 0.05, 0.1],
        'matrix': [0, 0],
        # 'rdm': [0.97, 0.92, 0.01],  # cosine similarity
        'rdm': [0.9, 0.8, 0.02],  # 1- wasserstein_distance
    },
    'smol-property_prediction-esol': {
        'structure': [0.4, 0.05, 0.1],
    },
    'smol-property_prediction-lipo': {
        'structure': [0.6, 0.05, 0.1],
    }
}


def clean_target_text(text, task):
    if task in REGRESSION_BENCHMARKS:
        for token in ['<|', '|>', '<FLOAT>', '</FLOAT>', '</s>']:
            text = text.replace(token, '')
        target = float(text.strip())
    elif task in CLASSIFICATION_BENCHMARKS:
        for token in ['<BOOLEAN>', '</BOOLEAN>', '</s>']:
            text = text.replace(token, '')
        text = text.strip()
        target = 1 if text == 'True' else 0
    elif task in REACTION_BENCHMARKS:
        raise NotImplementedError(f"{args.task} is not supported for clean_target_text.")
    elif task in MOL2TEXT_BENCHMARKS:
        raise NotImplementedError(f"{args.task} is not supported for clean_target_text.")
    else:
        raise NotImplementedError(f"{args.task} is not supported for clean_target_text.")
    return target

def clean_smiles_text(selfies_text):
    selfies_pattern = re.compile(r"<SELFIES>\s*(.*?)\s*</SELFIES>")
    selfies_string = selfies_pattern.search(selfies_text).group(1)    
    return sf.decoder(selfies_string)

from multiprocessing import Pool, cpu_count
import multiprocessing

def group_by_target(dataset, target_range, num_procs=10):
    def process_data(start_idx, end_idx, dataset, target_range, result_list):
        local_results = []
        for idx in range(start_idx, end_idx):
            data = dataset[idx]
            value = clean_target_text(data['target_text'], )
            group_idx = bisect.bisect_left(target_range, value) - 1
            local_results.append((group_idx, idx))
        result_list.extend(local_results)

    chunk_size = len(dataset) // num_procs
    manager = multiprocessing.Manager()
    result_list = manager.list()
    processes = []

    for i in range(num_procs):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_procs - 1 else len(dataset)
        p = multiprocessing.Process(target=process_data, args=(start_idx, end_idx, dataset, target_range, result_list))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    indicies_by_target = [[] for _ in range(len(target_range) - 1)]
    for group_idx, idx in result_list:
        if 0 <= group_idx < len(indicies_by_target):
            indicies_by_target[group_idx].append(idx)

    return indicies_by_target

def tanimoto_similarity(smiles1, smiles2):
    fpgen = AllChem.GetRDKitFPGenerator()
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)
    return TanimotoSimilarity(fp1, fp2)


def structural_similarity_map(data, dataset, indicies_by_target, target_range, args):
    n_reject = args.n_reject
    threshold, min_threshold, th_step = task_threshold[args.task]['structure']
    smiles = clean_smiles_text(data['input_mol_string'])
    if args.task in REGRESSION_BENCHMARKS:
        group_idx = bisect.bisect_left(target_range, clean_target_text(data['target_text'])) - 1
        
        far = len(target_range) // 3
        
        candi_indicies = list(chain.from_iterable(
            sublist for i, sublist in enumerate(indicies_by_target) 
            if i < group_idx - far or group_idx + far < i
        ))
    elif args.task in CLASSIFICATION_BENCHMARKS:
        # 0 : False, 1 : True
        candi_indicies = indicies_by_target[0] if clean_target_text(data['target_text'], args.task) == 1 \
                            else indicies_by_target[1]
    else:
        raise NotImplementedError(f"{args.task} is not supported for reject molecule generation.")

    used_indicies = []

    for i in range(n_reject):
        try_count = 0
        while True:
            candi_idx = random.choice(candi_indicies)
            if dataset[candi_idx]['input_mol_string'] == data['input_mol_string']:
                continue
            elif candi_idx in used_indicies:
                continue
            sim_score = tanimoto_similarity(smiles, clean_smiles_text(dataset[candi_idx]['input_mol_string']))
            if sim_score > threshold:
                rejected_data = dataset[candi_idx]
                
                data[f"{i}-th_rejected_selfies"] = sf.encoder(clean_smiles_text(dataset[candi_idx]['input_mol_string']))
                
                data[f"{i}-th_rejected_x"] = rejected_data["x"]
                data[f"{i}-th_rejected_edge_index"] = rejected_data["edge_index"]
                data[f"{i}-th_rejected_edge_attr"] = rejected_data["edge_attr"]
                data[f"{i}-th_additional_rejected_x"] = data['additional_x']
                data[f"{i}-th_additional_rejected_edge_index"] = data['additional_edge_index']
                data[f"{i}-th_additional_rejected_edge_attr"] = data['additional_edge_attr']
                used_indicies.append(candi_idx)
                break

            if try_count > 100:
                threshold -= th_step
                try_count = 0
            if threshold < min_threshold:  # if reached min threshold, randomly select (easy negative)
                threshold = -10.0
            
            try_count += 1
    return data


def match_noon_size(noon1, noon2):  # crop or padding
    min_size = min(len(noon1), len(noon2))
    noon1 = noon1[:min_size]
    noon2 = noon2[:min_size]
    return noon1, noon2

def get_rdm_similarity(noon1, noon2, method='cosine'):
    # output is post-processed to make large values are consistently better
    if method == 'euclidean':
        return -1 * euclidean(noon1, noon2) 
    elif method == 'cosine':
        return 1 - cosine(noon1, noon2)
    elif method == 'wasserstein':
        from scipy.stats import wasserstein_distance
        return 1 - wasserstein_distance(noon1, noon2)
    
def rdm_similarity_map(data, dataset, indicies_by_target, target_range, args):
    n_reject = args.n_reject
    threshold, min_threshold, th_step = task_threshold[args.task][args.reject_method]
    smiles = clean_smiles_text(data['input_mol_string'])
    group_idx = bisect.bisect_left(target_range, clean_target_text(data['target_text'])) - 1

    candi_indicies = list(chain.from_iterable(
        sublist for i, sublist in enumerate(indicies_by_target) if i != group_idx
    ))

    used_indicies = []
    init_threshold = threshold
    total_try_count = 0
    noon1 = data['nonn']
    for i in range(n_reject):
        try_count = 0
        while True:
            candi_idx = random.choice(candi_indicies)
            if dataset[candi_idx]['input_mol_string'] == data['input_mol_string']:
                continue
            elif candi_idx in used_indicies:
                continue

            if data['dft_result']['symbols'][0] == 'E' or (len(noon1) == 5 and np.allclose(noon1, np.array([0.0] * 5))):  # DFT is not succussed. Randomly select
                sim_score = 9999
            else:
                noon2 = dataset[candi_idx]['nonn']
                if len(noon2) == 5 and np.allclose(noon2, np.array([0.0] * 5)):    
                    if threshold >= min_threshold:
                        continue

                noon1, noon2 = match_noon_size(noon1, noon2)
                
                # sim_score = get_rdm_similarity(noon1, noon2, method='cosine')
                sim_score = get_rdm_similarity(noon1, noon2, method='wasserstein')
            
            # print(f"sim_score: {sim_score}, threshold: {threshold}, index: {i}, try: {try_count}, total: {total_try_count}")    
            if sim_score > threshold:
                rejected_data = dataset[candi_idx]
                
                data[f"{i}-th_rejected_selfies"] = sf.encoder(clean_smiles_text(dataset[candi_idx]['input_mol_string']))
                
                data[f"{i}-th_rejected_x"] = rejected_data["x"]
                data[f"{i}-th_rejected_edge_index"] = rejected_data["edge_index"]
                data[f"{i}-th_rejected_edge_attr"] = rejected_data["edge_attr"]
                data[f"{i}-th_additional_rejected_x"] = data['additional_x']
                data[f"{i}-th_additional_rejected_edge_index"] = data['additional_edge_index']
                data[f"{i}-th_additional_rejected_edge_attr"] = data['additional_edge_attr']
                used_indicies.append(candi_idx)
                break
            # if try_count > 50:
            #     threshold = max(threshold - th_step, min_threshold)
            #     try_count = 0
            # if total_try_count > 200:
            #     threshold = min_threshold
            # if total_try_count > 1000:
            #     threshold = 0.5
            if try_count > 100:
                threshold -= th_step
                try_count = 0
            if threshold < min_threshold:  # if reached min threshold, randomly select (easy negative)
                threshold = -10.0
                
            try_count += 1
            total_try_count += 1
    return data
    

def main(args):
    print(f"Start creating reject molecule for {args.task}")

    dataset_path = os.path.join(args.dataset_dir, args.dataset_path)
    result_path = os.path.join(dataset_path, args.reject_method)
    os.makedirs(result_path, exist_ok=True)

    dataset = load_from_disk(dataset_path)
    # print(dataset[0].keys())
    
    print(f"Loaded dataset with {len(dataset)} samples.")
    print(f"Start creating reject molecule by {args.reject_method}")

    range_cfg = task2range[args.task]
    if args.task in REGRESSION_BENCHMARKS:
        target_range = np.arange(range_cfg['start_value'], range_cfg['end_value'] + range_cfg['step_size'], range_cfg['step_size'])
        indicies_by_target = group_by_target(dataset, target_range)
        print(f"Target range: {len(target_range)}, {target_range}")
    elif args.task in CLASSIFICATION_BENCHMARKS:
        target_range = None
        indicies_by_target = [[] for _ in range(2)]
        for idx, data in enumerate(dataset):
            value = clean_target_text(data['target_text'], args.task)
            group_idx = value  # 0 : False, 1 : True
            indicies_by_target[group_idx].append(idx)
    else:
        raise NotImplementedError(f"{args.task} is not supported for reject molecule generation.")
    
    

    if args.reject_method == 'structure':
        reject_method = structural_similarity_map
    elif args.reject_method == 'matrix':
        reject_method = matrix_similarity_map
    elif args.reject_method == 'rdm':
        reject_method = rdm_similarity_map
    else:
        raise NotImplementedError(f'{args.reject_method} is not supported.')
    
    wrapped_func = partial(
        reject_method,
        dataset=dataset,
        indicies_by_target=indicies_by_target,
        target_range=target_range,
        args=args
    )

    print("Starting multiprocessing map...")
    new_dataset = dataset.map(
        wrapped_func,
        num_proc=args.num_procs,
        desc="Generating rejected samples"
    )

    new_dataset.save_to_disk(result_path)
    print(f"Saved new dataset with rejected molecules to {result_path}")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/data/data/Mol-LLM-v7.1', help="Path to the dataset directory")
    parser.add_argument('--dataset_path', type=str, default='mistralai-Mistral-7B-Instruct-v0.3_string+graph_q32_train_qm9_homo_0219', help="Path to the dataset")
    
    parser.add_argument("--task", type=str, default='qm9_homo', help="Task name")
    
    parser.add_argument("--n_reject", type=int, default=12, help="Number of rejected molecules to find")
    parser.add_argument('--reject_method', type=str, default='structure', choices=['structure', 'matrix', 'rdm'])
    
    parser.add_argument("--num_procs", type=int, default=32, help="Number of processes for multiprocessing")
    args = parser.parse_args()

    #tasks = ['smol-property_prediction-clintox', 'smol-property_prediction-hiv', 'smol-property_prediction-clintox']
    tasks = []
    
    for task in tasks:
        
        args.task = task
        print(args.task)
        main(args)

    main(args)
