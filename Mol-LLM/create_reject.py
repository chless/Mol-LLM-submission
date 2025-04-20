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


def clean_target_text(text):
    for token in ['<|', '|>', '<FLOAT>', '</FLOAT>', '</s>']:
        text = text.replace(token, '')
    return float(text.strip())

def clean_smiles_text(selfies_text):
    selfies_pattern = re.compile(r"<SELFIES>\s*(.*?)\s*</SELFIES>")
    selfies_string = selfies_pattern.search(selfies_text).group(1)    
    return sf.decoder(selfies_string)

def group_by_target(dataset, target_range):
    indicies_by_target = [[] for _ in range(len(target_range) - 1)]
    for idx, data in enumerate(dataset):
        value = clean_target_text(data['target_text'])
        group_idx = bisect.bisect_left(target_range, value) - 1
        if 0 <= group_idx < len(indicies_by_target):
            indicies_by_target[group_idx].append(idx)
        else:
            indicies_by_target[-1].append(idx)
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
    group_idx = bisect.bisect_left(target_range, clean_target_text(data['target_text'])) - 1

    # candi_indicies = list(chain.from_iterable(
    #     sublist for i, sublist in enumerate(indicies_by_target) if i != group_idx
    # ))
    far = len(target_range) // 3
    candi_indicies = list(chain.from_iterable(
        sublist for i, sublist in enumerate(indicies_by_target) 
        if i < group_idx - far or group_idx + far < i
    ))

    used_indicies = []
    init_threshold = threshold
    total_try_count = 0

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
            # if try_count > 50:
            #     threshold = max(threshold - th_step, min_threshold)
            #     try_count = 0
            # if total_try_count > 200:
            #     threshold = min_threshold
            # if total_try_count > 1000:
            #     threshold = 0.0
            # if try_count > 50:
            #     threshold -= th_step
            #     try_count = 0
            # if threshold < min_threshold:  # if reached min threshold, randomly select (easy negative)
            #     threshold = -10.0
            if try_count > 100:
                threshold -= th_step
                try_count = 0
            if threshold < min_threshold:  # if reached min threshold, randomly select (easy negative)
                threshold = -10.0
            
            try_count += 1
            total_try_count += 1
    return data


def match_noon_size(noon1, noon2):  # crop or padding
    # max_size = max(len(noon1), len(noon2))
    # noon1 = np.pad(noon1, (0, max_size - len(noon1)), constant_values=0.0)
    # noon2 = np.pad(noon2, (0, max_size - len(noon2)), constant_values=0.0)
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

    args.dataset_path = os.path.join(args.dataset_path, args.split, args.task)
    
    args.result_path = os.path.join(args.result_path, args.split, args.reject_method, args.task)
    os.makedirs(args.result_path, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    # print(dataset[0].keys())
    
    print(f"Loaded dataset with {len(dataset)} samples.")
    print(f"Start creating reject molecule by {args.reject_method}")

    range_cfg = task2range[args.task]
    target_range = np.arange(range_cfg['start_value'], range_cfg['end_value'] + range_cfg['step_size'], range_cfg['step_size'])
    indicies_by_target = group_by_target(dataset, target_range)

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
        num_proc=args.num_proc,
        desc="Generating rejected samples"
    )

    new_dataset.save_to_disk(args.result_path)
    print(f"Saved new dataset with rejected molecules to {args.result_path}")
    
    
    # args.dataset_path = os.path.join(args.dataset_path, args.split, args.task)
    # # args.result_path = os.path.join(args.result_path, args.split, args.reject_method, args.task)
    # args.result_path = os.path.join(args.result_path, args.split, args.reject_method)
    # os.makedirs(args.result_path, exist_ok=True)
    
    # dataset = load_from_disk(args.dataset_path)
    
    # args.result_path = os.path.join(args.result_path, args.task)
    # n_split = 10
    # split_size = len(dataset) // n_split
    # for i in range(n_split):
        
    #     save_path = args.result_path + f'{i}'
        
    #     start_idx = split_size * i
    #     end_idx = split_size * (i + 1) if i < n_split - 1 else len(dataset)  # 마지막 조각은 남은 전부
    #     split_dataset = dataset.select(np.arange(start_idx, end_idx))
    #     # split_dataset = dataset.select(np.arange(split_size * i , split_size*(i+1), 1))
    #     print(f"Processing split {i + 1}/{n_split}...")
    #     print(f"Loaded dataset with {len(split_dataset)} samples.")

    #     range_cfg = task2range[args.task]
    #     target_range = np.arange(range_cfg['start_value'], range_cfg['end_value'] + range_cfg['step_size'], range_cfg['step_size'])
    #     indicies_by_target = group_by_target(dataset, target_range)

    #     if args.reject_method == 'structure':
    #         reject_method = structural_similarity_map
    #     elif args.reject_method == 'matrix':
    #         reject_method = matrix_similarity_map
    #     elif args.reject_method == 'rdm':
    #         reject_method = rdm_similarity_map
    #     else:
    #         raise NotImplementError(f'{args.reject_method} is not supported.')
        
    #     # 함수 wrapping
    #     wrapped_func = partial(
    #         # structural_similarity_map,
    #         reject_method,
    #         dataset=dataset,
    #         indicies_by_target=indicies_by_target,
    #         target_range=target_range,
    #         args=args
    #     )

    #     print("Starting multiprocessing map...")
    #     new_dataset = split_dataset.map(
    #         wrapped_func,
    #         num_proc=args.num_proc,
    #         desc="Generating rejected samples"
    #     )

    #     new_dataset.save_to_disk(save_path)
    #     print(f"Saved new dataset with rejected molecules to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='/data/data/nonn', help="Path to the dataset")
    parser.add_argument("--result_path", type=str, default='/data/data/smilar_data', help="Path to save the result")
    
    parser.add_argument("--split", type=str, default='test', help="Dataset split")
    parser.add_argument("--task", type=str, default='qm9_homo', help="Task name")
    
    parser.add_argument("--n_reject", type=int, default=12, help="Number of rejected molecules to find")
    parser.add_argument('--reject_method', type=str, default='rdm', choices=['structure', 'matrix', 'rdm'])
    
    parser.add_argument("--num_proc", type=int, default=32, help="Number of processes for multiprocessing")
    args = parser.parse_args()

    
    main(args)
