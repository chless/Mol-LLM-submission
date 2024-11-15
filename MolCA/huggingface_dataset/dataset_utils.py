


import torch

import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater
from transformers import Trainer, DataCollatorForSeq2Seq

import re

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None, return_tensors=None, use_graph=False):
        super().__init__(tokenizer, padding=padding, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors)
        self.use_graph = use_graph
        # self.tokenizer = tokenizer
        
        if self.use_graph:
            # Collater with no special follow_batch or exclude_keys
            self.graph_collator = GraphCollater([], [])
    
    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        input_ids = [sample['input_ids'] for sample in batch]
        attention_mask = [sample['attention_mask'] for sample in batch]
        labels = [sample.pop('labels') for sample in batch]
        tasks = [task2id(sample.pop('task')) for sample in batch]
        
        
        if self.use_graph:
            graph_data_list = [
                Data(x=torch.tensor(sample['x'], dtype=torch.int64),
                     edge_index=torch.tensor(sample['edge_index'], dtype=torch.int64),
                     edge_attr=torch.tensor(sample['edge_attr'], dtype=torch.int64)
                     ) for sample in batch]

        import time
        start_time = time.time()
        features = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        max_length = features['input_ids'].shape[1]
        
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            if self.tokenizer.padding_side == "right":
                padded_label = label + [-100] * padding_length
            else:
                padded_label = [-100] * padding_length + label
            padded_labels.append(padded_label)
        
        features['labels'] = torch.tensor(padded_labels, dtype=torch.int64)
        
        # features['task'] = torch.tensor(tasks, dtype=torch.int8)
        features['task'] = tasks
        if self.use_graph:
            graphs = self.graph_collator(graph_data_list)
            features['graphs'] = graphs
            features['is_mol_token'] = (torch.tensor(features['input_ids']) == self.tokenizer.mol_token_id)
        return features




def generate_and_tokenize_prompt(data_point, tokenizer, args, test=False):
    default_system_prompt = 'You are a helpful assistant for molecular chemistry, \
to address tasks including molecular property classification, \
molecular property regression, chemical reaction prediction, \
molecule captioning, molecule generation. \n\n'

    input_text = default_system_prompt + data_point['input']
    output_text = data_point['output']
    
    input_mol_string_pattern = re.compile(
            "<SELFIES>"
            + ".*?"
            + "</SELFIES>"
            # + r"(?=.*\[/INST\])"
        )
    def postfix_graph_sequence(match):
        return match.group(0) + graph_sequence

    graph_sequence = ("<GRAPH>" + "<mol>" * args.num_query_token + "</GRAPH>")
    
    if args.mol_representation == 'graph_only':
        input_text = input_mol_string_pattern.sub(
            graph_sequence, 
            input_text
            )
    elif args.mol_representation == 'string+graph':
        input_text = input_mol_string_pattern.sub(
            postfix_graph_sequence, 
            input_text
            )
    
    prompt_tokenized = tokenizer(
                            tokenizer.bos_token + ' ' + input_text,
                            truncation=True,
                            max_length=args.cutoff_len,
                            padding=False,
                            return_tensors=None,
                            add_special_tokens=False,
                            )
    target_tokenized = tokenizer(
                            output_text + ' ' + tokenizer.eos_token,
                            max_length=args.cutoff_len - len(prompt_tokenized['input_ids']),
                            padding=False,
                            return_tensors=None,
                            add_special_tokens=False,
                            )
        
    input_ids = prompt_tokenized['input_ids'] + target_tokenized['input_ids']
    attention_mask = prompt_tokenized['attention_mask'] + target_tokenized['attention_mask']
    labels = input_ids.copy()
    prompt_length = len(prompt_tokenized['input_ids'])
    labels[:prompt_length] = [-100] * prompt_length  # masking input
    if test:
        input_ids = input_ids[:prompt_length]
        attention_mask = attention_mask[:prompt_length]

    tokenized_result ={
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,  
        
    }
    
    return tokenized_result



def task2id(task):
    task2id = {
            # name conversion
            'smol-name_conversion-i2s': 0,
            'smol-name_conversion-i2f': 1,
            'smol-name_conversion-s2f': 2,
            'smol-name_conversion-s2i': 3,
            
            # propertry classification
            'smol-property_prediction-bbbp': 4,
            'smol-property_prediction-clintox': 5,
            'smol-property_prediction-hiv': 6,
            'smol-property_prediction-sider': 7,
            'bace': 8,
            'tox21': 9,
            'toxcast': 10,
            
            # propertry regression
            'smol-property_prediction-esol': 11,
            'smol-property_prediction-lipo': 12,
            'qm9_homo': 13,
            'qm9_lumo': 14,
            'qm9_homo_lumo_gap': 15,
            'qm9_dipole_moment': 16,
            'qm9_isotropic_polarizability': 17,
            'qm9_electronic_spatial_extent': 18,
            'qm9_zero_point_vibrational_energy': 19,
            'qm9_heat_capacity_298K': 20,
            'qm9_internal_energy_298K': 21,
            'qm9_enthalpy_298K': 22,
            'qm9_free_energy_298K': 23,
            
            'forward_reaction_prediction': 24,
            'smol-forward_synthesis': 25,
            'retrosynthesis': 26,
            'smol-retrosynthesis': 27,
            'reagent_prediction': 28,
            
            'chebi-20-text2mol': 29,
            'smol-molecule_generation': 30,
            'chebi-20-mol2text': 31,
            'smol-molecule_captioning': 32,
    }
    return task2id[task]

def id2task(task_id):
    # task id to task name
    id2task = {
        # name conversion
        0: 'smol-name_conversion-i2s',
        1: 'smol-name_conversion-i2f',
        2: 'smol-name_conversion-s2f',
        3: 'smol-name_conversion-s2i',
    
        # propertry classification
        4: 'smol-property_prediction-bbbp',
        5: 'smol-property_prediction-clintox',
        6: 'smol-property_prediction-hiv',
        7: 'smol-property_prediction-sider',
        8: 'bace',
        9: 'tox21',
        10: 'toxcast',
    
        # propertry regression
        11: 'smol-property_prediction-esol',
        12: 'smol-property_prediction-lipo',
        13: 'qm9_homo',
        14: 'qm9_lumo',
        15: 'qm9_homo_lumo_gap',
        16: 'qm9_dipole_moment',
        17: 'qm9_isotropic_polarizability',
        18: 'qm9_electronic_spatial_extent',
        19: 'qm9_zero_point_vibrational_energy',
        20: 'qm9_heat_capacity_298K',
        21: 'qm9_internal_energy_298K',
        22: 'qm9_enthalpy_298K',
        23: 'qm9_free_energy_298K',
    
        24: 'forward_reaction_prediction',
        25: 'smol-forward_synthesis',
        26: 'retrosynthesis',
        27: 'smol-retrosynthesis',
        28: 'reagent_prediction',
        
        29: 'chebi-20-text2mol',
        30: 'smol-molecule_generation',
        31: 'chebi-20-mol2text',
        32: 'smol-molecule_captioning',
            
        }
    return id2task[task_id]