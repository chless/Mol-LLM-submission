import os
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

import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys



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

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
            # llm_model, use_fast=False, padding_side="right"
            model_name, use_fast=False, padding_side="left"
        )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "\n"})

    # if self.args.add_selfies_tokens:
    if True:
        # Read txt from selfies_token_path
        with open('/text-mol/Mol-LLM/model/selfies_dict.txt', "r") as f:
            selfies_tokens = f.readlines()
            selfies_tokens = [token.strip() for token in selfies_tokens]
        tokenizer.add_tokens(selfies_tokens)
        # get token id of the selfies_tokens
        tokenizer.selfies_token_ids = [
            tokenizer(token, add_special_tokens=False).input_ids[0]
            for token in selfies_tokens
        ]
        tokenizer.added_selfies_tokens = selfies_tokens
        # remove '.' from the marked list for selfies token
        # tokenizer.added_selfies_tokens.remove(".")
        # tokenizer.selfies_token_ids.remove(36)
        print(f"Added {len(selfies_tokens)} selfies tokens to the tokenizer")

    additional_tokens = [
                getattr(added_tokens, tokens)
                for tokens in dir(added_tokens)
                if not re.match("__.*__", tokens)
            ]
    additional_tokens = [
        token for sublist in additional_tokens for token in sublist
    ]

    tokenizer.add_tokens(additional_tokens)

    # tokenizer.mol_token = added_tokens.MOL_EMBEDDING[0]
    tokenizer.add_special_tokens({"additional_special_tokens": [added_tokens.MOL_EMBEDDING[0]]})
    tokenizer.mol_token_id = tokenizer.convert_tokens_to_ids(added_tokens.MOL_EMBEDDING[0])
    return tokenizer

def tanimoto_similarity(smiles1, smiles2):
    fpgen = AllChem.GetRDKitFPGenerator()
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)
    return TanimotoSimilarity(fp1, fp2)

def fingerprint_similarity(smiles1, smiles2, method='tanimoto'):
    if method == 'tanimoto':
        return tanimoto_similarity(smiles1, smiles2)
    else:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if method == 'maccs':
            return DataStructs.FingerprintSimilarity(
                        MACCSkeys.GenMACCSKeys(mol1),
                        MACCSkeys.GenMACCSKeys(mol2),
                        metric=DataStructs.TanimotoSimilarity)
        elif method == 'rdk':
            return DataStructs.FingerprintSimilarity(
                        Chem.RDKFingerprint(mol1),
                        Chem.RDKFingerprint(mol2),
                        metric=DataStructs.TanimotoSimilarity)
        elif method == 'morgan':
            morgan_r = 2
            return DataStructs.TanimotoSimilarity(
                        AllChem.GetMorganFingerprint(mol1, morgan_r),
                        AllChem.GetMorganFingerprint(mol2, morgan_r),
                        metric=DataStructs.TanimotoSimilarity)
        else:
            raise NotImplementedError(f"{method} is not supported for fingerprint similarity.")

def text_similarity(text1, text2, tokenizer, method='bleu'):
    if method in ['bleu', 'meteor']:
        in_tok = tokenizer.tokenize(text1)
        out_tok = tokenizer.tokenize(text2)
        if method == 'bleu':
            return corpus_bleu([[in_tok]], [out_tok], weights=(0.25, 0.25, 0.25, 0.25))
        elif method == 'meteor':
            return meteor_score([in_tok], out_tok) * 100
            
    elif method in ['rouge1', 'rouge2', 'rougeL']:
        scorer = rouge_scorer.RougeScorer([method], use_stemmer=True)
        # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(text1, text2)
        return scores[method].fmeasure
    else:
        raise NotImplementedError(f"{method} is not supported for text similarity.")






# 1-RDM based similarity

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