# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from datasets import load_dataset, load_from_disk, load_dataset_builder

from data_utils import DataCollator, generate_and_tokenize_prompt


class Stage3DM(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        mode: str = "to_be_removed",
        num_workers: int = 0,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.num_workers = num_workers

        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.inference_batch_size = args.inference_batch_size
        self.inference_max_length = args.inference_max_length

        self.mol_representation = args.mol_representation
        self.tokenizer = tokenizer
        
        self.train_dataset = get_dataset('train', tokenizer, args)
        self.val_dataset = get_dataset('validation', tokenizer, args)
        self.test_dataset = get_dataset('test', tokenizer, args)
        
        builder = load_dataset_builder(os.path.join(args.raw_data_root, 'InstructGraph.py'))
        builder.config.train_tasks
        self.task_subtask_name_pairs = list(builder.config.test_tasks)

        tokenizer.padding_side = "left"
        self.train_collator = DataCollator(
            tokenizer=tokenizer,
            # pad_to_multiple_of=8,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
            use_graph='graph' in self.mol_representation,
        )
        self.eval_collator = DataCollator(
            tokenizer=tokenizer,
            # pad_to_multiple_of=8,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
            use_graph='graph' in self.mol_representation,
            train=False,
        )
        

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.train_collator
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.eval_collator
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.eval_collator
        )
        return loader





def get_dataset(split, tokenizer, args):
    
    data_path = args.raw_data_root
    mol_representation = args.mol_representation
    num_query_token = args.num_query_token
    base_model = args.llm_model.replace('/', '-')
    
    # TODO add filtering specific tasks
    if args.tasks is not None and len(args.tasks) == 0:
        tasks = None
    else:
        tasks = args.tasks
    
    # to avoid re-generating
    if 'graph' in mol_representation:
        preprocessed_data_path = os.path.join(data_path, f"preprocessd_{base_model}_{split}_{mol_representation}_{num_query_token}")
    else:
        preprocessed_data_path = os.path.join(data_path, f"preprocessd_{base_model}_{split}_{mol_representation}")
    print("preprocessed_data_path:", preprocessed_data_path)
    
    if os.path.exists(preprocessed_data_path):
        dataset = load_from_disk(preprocessed_data_path)
    else:  # preprocess data
        
        dataset = load_dataset(
            path=os.path.join(data_path, 'InstructGraph.py'),
            split=split, 
            tasks=tasks,
            cache_dir=os.path.join(data_path, 'cache'),
            )
        
        # when you debug, you can use this line to reduce the dataset size
        # dataset = dataset.select(torch.randperm(len(dataset))[:1000])
        
        remove_keys = set(dataset.column_names)
        remove_keys -= {
                'task', 
                'x', 'edge_index', 'edge_attr', 
                'additional_x', 'additional_edge_index', 'additional_edge_attr'
            }
        dataset = dataset.shuffle().map(
                        generate_and_tokenize_prompt,
                        remove_columns=remove_keys,
                        fn_kwargs={
                            'tokenizer': tokenizer, 
                            'args': args, 
                            'test': False if split=='train' else True
                            },
                        
                        load_from_cache_file=False,
                        )
        # save preprocessd data
        dataset.save_to_disk(preprocessed_data_path)
    
    return dataset
