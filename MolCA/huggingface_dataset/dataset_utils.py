


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
            graphs = [
                Data(x=torch.tensor(sample['x'], dtype=torch.int64),
                     edge_index=torch.tensor(sample['edge_index'], dtype=torch.int64),
                     edge_attr=torch.tensor(sample['edge_attr'], dtype=torch.int64)
                     ) for sample in batch]
            additional_graphs = [
                Data(x=torch.tensor(sample['additional_x'], dtype=torch.int64),
                     edge_index=torch.tensor(sample['additional_edge_index'], dtype=torch.int64),
                     edge_attr=torch.tensor(sample['additional_edge_attr'], dtype=torch.int64)
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
        
        features['task'] = tasks
        if self.use_graph:
            graphs = self.graph_collator(graphs)
            additional_graphs = self.graph_collator(additional_graphs)
            features['graphs'] = graphs
            features['additional_graphs'] = additional_graphs
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



tasks = [
    # name conversion
    'smol-name_conversion-i2s',
    'smol-name_conversion-i2f',
    'smol-name_conversion-s2f',
    'smol-name_conversion-s2i',
    
    # propertry classification
    'smol-property_prediction-bbbp',
    'smol-property_prediction-clintox',
    'smol-property_prediction-hiv',
    'smol-property_prediction-sider',
    'bace',
    'tox21',
    'toxcast',
    
    # propertry regression
    'smol-property_prediction-esol',
    'smol-property_prediction-lipo',
    'qm9_homo',
    'qm9_lumo',
    'qm9_homo_lumo_gap',
    'qm9_dipole_moment',
    'qm9_isotropic_polarizability',
    'qm9_electronic_spatial_extent',
    'qm9_zero_point_vibrational_energy',
    'qm9_heat_capacity_298K',
    'qm9_internal_energy_298K',
    'qm9_enthalpy_298K',
    'qm9_free_energy_298K',
    
    'forward_reaction_prediction',
    'smol-forward_synthesis',
    'retrosynthesis',
    'smol-retrosynthesis',
    'reagent_prediction',
    
    'chebi-20-text2mol',
    'smol-molecule_generation',
    'chebi-20-mol2text',
    'smol-molecule_captioning',
]


def task2id(task):
    task2id = {k: i for i, k in enumerate(tasks)}
    return task2id[task]

def id2task(task_id):
    id2task = {i: k for i, k in enumerate(tasks)}
    return id2task[task_id]