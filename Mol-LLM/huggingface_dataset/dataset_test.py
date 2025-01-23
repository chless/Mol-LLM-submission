import os
import sys
from typing import List
import numpy as np
import random
import fire
import torch
import transformers
from datasets import load_dataset, load_from_disk


import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

from dataset_utils import CustomDataCollator, generate_and_tokenize_prompt, task2id, id2task

from easydict import EasyDict

def set_random_seeds(seed: int = 13):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seeds()


def train(
    # model/data params
    base_model: str = "mistralai/Mistral-7B-v0.1", 
    data_path: str = "",
    # training hyperparams
    batch_size: int = 512,
    micro_batch_size: int = 8,
    cutoff_len: int = 512,

    train_split='train',
    tasks: List[str] = None,
    
    debug: bool = False,
    
    # for Mol-LLM
    mol_representation: str = "string_only",
    num_query_token: int = 4,
    
    num_workers=0,
):
   
    # gradient_accumulation_steps = batch_size // micro_batch_size

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "\n"})
        
    mol_token = "<mol>"
    tokenizer.add_special_tokens({"additional_special_tokens": [mol_token]})
    mol_token_id = tokenizer(mol_token, add_special_tokens=False)['input_ids'][0]
    tokenizer.mol_token_id = mol_token_id
    
    tokenizer.paddding_side = "left"
        



    
    if tasks is not None and len(tasks) == 0:
        tasks = None
    
    train_data = load_dataset(
        path=os.path.join(data_path, 'InstructGraph.py'),
        # data_path, 
        split=train_split, 
        tasks=tasks,
        cache_dir=os.path.join(data_path, 'cache'),
        # num_proc=8,
        )
    
    if debug:
        num_debug_samples = 10000
        print(f"Debugging mode: using only {num_debug_samples} samples")
        train_data = train_data.select(torch.randperm(len(train_data))[:num_debug_samples])
    
    remove_keys = set(train_data.column_names)
    remove_keys -= {'task'}
    if 'graph' in mol_representation:  # TODO can be removed
        remove_keys -= {'x', 'edge_index', 'edge_attr', 'additional_x', 'additional_edge_index', 'additional_edge_attr'}
    
    
    
    
    # it can be replaced with argparse later
    args = {
        'cutoff_len': cutoff_len,
        'num_query_token': num_query_token,
        'mol_representation': mol_representation,
    }
    args = EasyDict(args)
    
    
    # to avoid re-generating
    if 'graph' in mol_representation:
        preprocessed_data_path = os.path.join(data_path, f"preprocessd_{base_model.replace('/', '-')}_{train_split}_{mol_representation}_{num_query_token}")
    else:
        preprocessed_data_path = os.path.join(data_path, f"preprocessd_{base_model.replace('/', '-')}_{train_split}_{mol_representation}")
        
    if os.path.exists(preprocessed_data_path):
        train_data = load_from_disk(preprocessed_data_path)
    else:
        # preprocess data
        train_data = train_data.shuffle().map(generate_and_tokenize_prompt,
                                            remove_columns=remove_keys,
                                            fn_kwargs={
                                                'tokenizer': tokenizer, 
                                                'args': args, 
                                                'test': False  # if test is True, the prompts do not include target
                                                },
                                            )
        # save preprocessd data
        train_data.save_to_disk(preprocessed_data_path)

    print('len of train_data:', len(train_data))


    data_collator = CustomDataCollator(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
        use_graph='graph' in mol_representation,
    )
    
    
    data_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=micro_batch_size, 
        collate_fn=data_collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        )
    
    
    # for batch in tqdm(data_loader):
    #     """
    #     batch have the following:
    #         input_ids
    #         attention_mask
    #         labels
    #         task
    #     task is task_id, if you want to convert it to task name, use id2task
    #     [id2task(task_id) for task_id in batch.tasks]
    #     """
        
    #     pass
    
    # Trainer with DDP
    trainer = pl.Trainer(
        devices='0,1',
        strategy='ddp',
        max_epochs=1,
    )

    model = MyModel()
    trainer.fit(model, data_loader)



# Dummy LightningModule
import torch.nn as nn
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Adding a dummy parameter
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)



if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)



# CUDA_VISIBLE_DEVICES=0,1 python dataset_test.py --base_model="mistralai/Mistral-7B-v0.1" --data_path=/data/llasmol-all/Mol-LLM-v6
