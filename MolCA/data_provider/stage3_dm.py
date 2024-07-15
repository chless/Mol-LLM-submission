# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric

# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_caption_dataset import MoleculeCaption, MoleculeCaptionV2
import re
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import InMemoryDataset, Data
import os

import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
from torch.utils.data import ConcatDataset
from datasets import load_dataset
from data_provider import instructions
import numpy as np
import selfies
from tqdm import tqdm

# we split individual characters inside special tokens like [START_DNA]
# TODO: change this ugly I_SMILES things to regular special token, and add the special token to vocab whichever LLM
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.

BOOL_TOKENS = ["<BOOLEAN>", "</BOOLEAN>"]
FLOAT_TOKENS = ["<FLOAT>", "</FLOAT>"]
DESCRIPTION_TOKENS = ["<DESCRIPTION>", "</DESCRIPTION>"]


def smiles_handler(text, mol_ph, mol_representation, model="llama"):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    # graph embedding without smiles tokens
    # '<mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    if mol_representation == "graph_only":
        text = CUSTOM_SEQ_RE.sub(r"%s" % (mol_ph), text)
    # smiles tokens without graph embedding
    # \1, \4 corresponds to the special tokens for string (\2 is the special token, which included in the nest of \1)
    # \3 corresponds to the content of the smiles token
    # '<Molecule>[H]N([H])C(=O)C([H])([H])[H]</Molecule>'
    elif mol_representation == "string_only":
        text = CUSTOM_SEQ_RE.sub(r"\1\3\4", text)
    # smiles tokens with graph embedding
    # '<Molecule>[H]N([H])C(=O)C([H])([H])[H]</Molecule><mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    elif mol_representation == "string+graph":
        text = CUSTOM_SEQ_RE.sub(r"\1\3\4%s" % (mol_ph), text)
    # smiles tokens with graph tokens without special tokens
    # '[H]N([H])C(=O)C([H])([H])[H]<mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    else:
        text = CUSTOM_SEQ_RE.sub(r"\3%s" % (mol_ph), text)
    # for reagent prediction, double the mol_ph with >> to separate reactant and product
    if ">>" in text:
        text += ">>" + mol_ph
    return text, smiles_list


class TrainCollater:
    def __init__(
        self,
        tokenizer,
        text_max_len,
        mol_ph,
        mol_token_id,
        mol_representation=True,
        multi_task=False,
        model=None,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation
        self.multi_task = multi_task
        self.model = model

    def __call__(self, batch):
        # in multi-task, perdevice  batch size should be multiple of 4: classificaiton, regression, translation, reaction
        if self.multi_task:
            graphs, texts, smiles_prompt, tasks, instructions = zip(*batch)
        else:
            graphs, texts, smiles_prompt, tasks = zip(*batch)

        try:
            graphs = self.collater(graphs)
        except:
            print(graphs)
            raise
        # graphs = self.collater(graphs)

        ## deal with prompt
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.mol_representation, self.model)[0]
            for p in smiles_prompt
        ]

        for i in range(len(texts)):
            smiles_prompt[i] += " " + instructions[i]

        self.tokenizer.padding_side = "left"
        smiles_prompt_tokens = self.tokenizer(
            text=smiles_prompt,
            truncation=True,
            padding="longest",
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens["is_mol_token"] = is_mol_token

        self.tokenizer.padding_side = "right"
        text_tokens = self.tokenizer(
            text=texts,
            truncation=True,
            padding="longest",
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return graphs, smiles_prompt_tokens, text_tokens, tasks


class InferenceCollater:
    def __init__(
        self,
        tokenizer,
        text_max_len,
        mol_ph,
        mol_token_id,
        mol_representation=True,
        multi_task=False,
        model=None,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation
        self.multi_task = multi_task
        self.model = model

    def __call__(self, batch):
        if self.multi_task:
            graphs, texts, smiles_prompt, tasks, instructions = zip(*batch)
        else:
            graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.mol_representation, self.model)[0]
            for p in smiles_prompt
        ]

        for i in range(len(texts)):
            smiles_prompt[i] += " " + instructions[i]

        ## deal with prompt
        self.tokenizer.padding_side = "left"
        smiles_prompt_tokens = self.tokenizer(
            smiles_prompt,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.text_max_len,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
        )
        texts = self.tokenizer(
            text=texts,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.text_max_len,
            truncation=True,
            padding="longest",
            return_attention_mask=True,
        )

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens["is_mol_token"] = is_mol_token
        return graphs, smiles_prompt_tokens, texts, tasks


# binary classification
CLASSIFICATION_BENCHMARKS = [
    "bace",  # 1 task # molca, biot5+, instructmol
    "bbbp",  # 1 task # molca, biot5+, instructmol, llasmol
    "clintox",  # 2 tasks # molca, biot5+, llasmol
    "toxcast",  # 617 # molca
    "sider",  # 27 # molca, llasmol
    "tox21",  # 12 tasks # molca
    "hiv",  # 1 tasks # biot5+, instructmol, llasmol
]
REGRESSION_BENCHMARKS = [
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
    "esol",  # 1 task # llasmol
    "lipo",  # 1 task # llasmol
]

MOL2TEXT_BENCHMARKS = ["molecular_description_generation"]

TEXT2MOL_BENCHMARKS = [
    "description_guided_molecule_design",
]

REACTION_BENCHMARKS = [
    "reagent_prediction",
    "forward_reaction_prediction",
    "retrosynthesis",
]

INSTRUCTION_CLASSIFICATION = "\n Given the molecule, you should predict {task_name} property of the molecule. The answer should be in the form {label_tokens[0]}True{label_tokens[1]} or {label_tokens[0]}False{label_tokens[1]}."
INSTRUCTION_REGRESSION = "\n Given the molecule, you should predict {task_name} property of the molecule. The answer should be in the form {label_tokens[0]}x.xxxx{label_tokens[1]}."


class Stage3DM(LightningDataModule):
    def __init__(
        self,
        mode: str = "pretrain",
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = "data/",
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.debug = args.debug
        self.args = args
        self.root = root

        if root == "multi_task":
            # preprocess dataset and save
            # if self.args.get_raw_data:
            # self.get_raw_multi_task_dataset()

            self.concat_datasets = {
                task: {"train": None, "val": None, "test": None}
                for task in ["classification", "regression", "reaction", "translation"]
            }

            for task in ["classification", "regression", "reaction", "translation"]:
                for split in ["train", "val", "test"]:
                    self.concat_datasets[task][split] = InstructionInMemoryDataset(
                        root=self.args.raw_data_root,
                        filename=f"{task}_{split}",
                        prompt=self.prompt,
                        resize=2400 if split == "val" else None,
                        mol_string_conversion=self.args.mol_string_conversion,
                    )

        elif root in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            self.tasks, self.train_data, self.val_data, self.test_data = (
                self.get_dataset(root)
            )
            self.tasks = [f"{root}/{t}" for t in self.tasks]
            self.train_dataset = MoleculeNetDatasetDeepChem(
                data=self.train_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=self.debug,
            )
            self.val_dataset = MoleculeNetDatasetDeepChem(
                data=self.val_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=self.debug,
            )
            self.test_dataset = MoleculeNetDatasetDeepChem(
                data=self.test_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=self.debug,
            )
        else:
            self.pretrain_dataset = MoleculeCaptionV2(
                root + f"pretrain.pt", text_max_len, self.prompt, debug=self.debug
            )
            self.train_dataset = MoleculeCaptionV2(
                root + f"train.pt", text_max_len, self.prompt, debug=self.debug
            )
            self.val_dataset = MoleculeCaptionV2(
                root + f"valid.pt", text_max_len, self.prompt, debug=self.debug
            )
            self.test_dataset = MoleculeCaptionV2(
                root + f"test.pt", text_max_len, self.prompt, debug=self.debug
            )

        self.init_tokenizer(tokenizer)
        self.mol_ph_token = "<mol>" * self.args.num_query_token
        self.mol_representation = args.mol_representation

    def get_raw_multi_task_dataset(
        self,
    ):
        task_subtask_lists = {
            "reagent_prediction": [0],
            "forward_reaction_prediction": [0],
            "retrosynthesis": [0],
            "bace": [0],
            "bbbp": [0],
            "clintox": [0, 1],
            "toxcast": [0],
            "sider": [0],
            "tox21": [0],
            "hiv": [0],
            "qm9": [0],
            "esol": [0],
            "lipo": [0],
            "description_guided_molecule_design": [0],
            "molecular_description_generation": [0],
        }
        self.task_subtask_pairs = [
            (task, subtask)
            for task, subtasks in task_subtask_lists.items()
            for subtask in subtasks
        ]

        total_benchmarks = (
            REACTION_BENCHMARKS
            + MOL2TEXT_BENCHMARKS
            + TEXT2MOL_BENCHMARKS
            + CLASSIFICATION_BENCHMARKS
            + REGRESSION_BENCHMARKS
        )

        multi_task_datasets = {
            task_name: self.get_dataset(task_name)  # {task_name: [train, val, test]
            for task_name in total_benchmarks
        }

        self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
        for task_subtask_pair in tqdm(
            self.task_subtask_pairs, desc="Processing task_subtask_pairs"
        ):
            task_name = task_subtask_pair[0]
            subtasks = multi_task_datasets[task_name][0]
            subtask_idx = task_subtask_pair[1]
            task_subtask_pair = f"{task_name}/{subtasks[subtask_idx]}"

            data_split = multi_task_datasets[task_name][
                1:
            ]  # train_set, val_set, test_set

            if task_name in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
                train_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
                valid_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
                test_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
            elif task_name in REACTION_BENCHMARKS:
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
            elif task_name in MOL2TEXT_BENCHMARKS:
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
            elif task_name in TEXT2MOL_BENCHMARKS:
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )

            self.train_dataset.append(train_dataset)
            self.val_dataset.append(valid_dataset)
            self.test_dataset.append(test_dataset)

        # concat datasets from each subtask into large class [classification, regression, reaction prediction, translation]

        concat_datasets = {
            task: {"train": [], "val": [], "test": []}
            for task in ["classification", "regression", "reaction", "translation"]
        }

        for i in range(len(self.task_subtask_pairs)):
            task_subtask_pair = self.task_subtask_pairs[i]
            task_name = task_subtask_pair[0]
            if task_name in CLASSIFICATION_BENCHMARKS:
                concat_datasets["classification"]["train"].append(self.train_dataset[i])
                concat_datasets["classification"]["val"].append(self.val_dataset[i])
                concat_datasets["classification"]["test"].append(self.test_dataset[i])
            elif task_name in REGRESSION_BENCHMARKS:
                concat_datasets["regression"]["train"].append(self.train_dataset[i])
                concat_datasets["regression"]["val"].append(self.val_dataset[i])
                concat_datasets["regression"]["test"].append(self.test_dataset[i])
            elif task_name in MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS:
                concat_datasets["translation"]["train"].append(self.train_dataset[i])
                concat_datasets["translation"]["val"].append(self.val_dataset[i])
                concat_datasets["translation"]["test"].append(self.test_dataset[i])
            elif task_name in REACTION_BENCHMARKS:
                concat_datasets["reaction"]["train"].append(self.train_dataset[i])
                concat_datasets["reaction"]["val"].append(self.val_dataset[i])
                concat_datasets["reaction"]["test"].append(self.test_dataset[i])
            else:
                raise NotImplementedError

        os.makedirs(os.path.join(self.args.raw_data_root, "raw"), exist_ok=True)

        for task in ["classification", "regression", "reaction", "translation"]:
            for split in ["train", "val", "test"]:
                concat_dataset = ConcatDataset(concat_datasets[task][split])
                torch.save(
                    concat_dataset,
                    f"{self.args.raw_data_root}/raw/{task}_{split}",
                )

        return concat_datasets

    def get_dataset(self, root):
        base_path = f"dataset/{root}"
        os.makedirs(base_path, exist_ok=True)
        # get dataset from deepchem
        if root == "bace":
            loading_fn = dc.molnet.load_bace_classification
        elif root == "esol":
            loading_fn = dc.molnet.load_delaney
        elif root in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            loading_fn = getattr(dc.molnet, f"load_{root}")
        elif root in MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS:
            mol_instruction_dataset = load_dataset(
                "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
            )
            dataset = mol_instruction_dataset[root]
            train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
            split = train_dataset.train_test_split(test_size=0.1, shuffle=True)
            train_dataset, valid_dataset = split["train"], split["test"]

            test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
            tasks = [root]
        elif root in REACTION_BENCHMARKS:
            mol_instruction_dataset = load_dataset(
                "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
            )
            dataset = mol_instruction_dataset[root]
            train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
            split = train_dataset.train_test_split(test_size=0.1, shuffle=True)
            train_dataset, valid_dataset = split["train"], split["test"]

            test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
            tasks = [root]
        else:
            raise NotImplementedError

        if root in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            tasks, datasets, transformers = loading_fn(
                featurizer="Raw",
                splitter="scaffold",
                save_dir=base_path,
                data_dir=base_path,
                reload=True,
            )
            train_dataset, valid_dataset, test_dataset = datasets
        else:
            pass
        return tasks, train_dataset, valid_dataset, test_dataset

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # self.train_dataset.tokenizer = tokenizer
        # self.val_dataset.tokenizer = tokenizer
        # self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == "pretrain":
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
        elif self.mode == "ft":
            if self.root == "multi_task":
                loader = [
                    DataLoader(
                        self.concat_datasets[task]["train"],
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=True,
                        collate_fn=TrainCollater(
                            self.tokenizer,
                            self.text_max_len,
                            self.mol_ph_token,
                            self.mol_token_id,
                            self.mol_representation,
                            multi_task=True,
                            model=self.args.opt_model,
                        ),
                    )
                    for task in [
                        "classification",
                        "regression",
                        "reaction",
                        "translation",
                    ]
                ]
            else:
                loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=True,
                    collate_fn=TrainCollater(
                        self.tokenizer,
                        self.text_max_len,
                        self.mol_ph_token,
                        self.mol_token_id,
                        self.mol_representation,
                        model=self.args.opt_model,
                    ),
                )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        if self.root != "multi_task":
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.inference_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                collate_fn=InferenceCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
            return [val_loader, test_loader]
        else:
            loader = [
                DataLoader(
                    self.concat_datasets[task]["val"],
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=InferenceCollater(
                        self.tokenizer,
                        self.text_max_len,
                        self.mol_ph_token,
                        self.mol_token_id,
                        self.mol_representation,
                        multi_task=True,
                        model=self.args.opt_model,
                    ),
                )
                for task in ["classification", "regression", "reaction", "translation"]
            ]
            return loader

    def test_dataloader(self):
        if self.root != "multi_task":
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.inference_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                collate_fn=InferenceCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
            return loader
        else:
            loader = [
                DataLoader(
                    self.concat_datasets[task]["test"],
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=InferenceCollater(
                        self.tokenizer,
                        self.text_max_len,
                        self.mol_ph_token,
                        self.mol_token_id,
                        self.mol_representation,
                        multi_task=True,
                        model=self.args.opt_model,
                    ),
                )
                for task in ["classification", "regression", "reaction", "translation"]
            ]
            return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--inference_batch_size", type=int, default=4)
        parser.add_argument("--use_smiles", action="store_true", default=False)
        parser.add_argument("--root", type=str, default="data/PubChemDataset_v4")
        parser.add_argument("--text_max_len", type=int, default=128)
        parser.add_argument(
            "--prompt",
            type=str,
            default="[START_I_SMILES]{}[END_I_SMILES]",
        )
        parser.add_argument("--filtered_cid_path", type=str, default=None)

        # moleculenet dataset
        parser.add_argument("--subtask_idx", type=int, default=0)
        parser.add_argument(
            "--raw_data_root", type=str, default="MolCA/data/multi_task_dataset"
        )
        parser.add_argument(
            "--mol_string_conversion",
            choices=[False, "smiles2selfies", "selfies2smiles"],
            default=False,
        )

        return parent_parser


from tqdm import tqdm

from rdkit import Chem


def wrap_label(label, task, mol_special_tokens):

    if task in CLASSIFICATION_BENCHMARKS:
        label_tokens = BOOL_TOKENS
    elif task in REGRESSION_BENCHMARKS:
        label_tokens = FLOAT_TOKENS
    elif task in MOL2TEXT_BENCHMARKS:
        label_tokens = DESCRIPTION_TOKENS
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        label_tokens = mol_special_tokens
    else:
        raise NotImplementedError

    if task in CLASSIFICATION_BENCHMARKS:
        if label:
            return label_tokens[0] + "True" + label_tokens[1]
        else:
            return label_tokens[0] + "False" + label_tokens[1]
    elif task in REGRESSION_BENCHMARKS:
        if isinstance(label, float):
            label = "{:.4f}".format(label)
        # force to predict the sign of label first
        if "-" not in label:
            label = "+" + label
        # unify the length of label to 7
        label = label[:7]
        while len(label) < 7:
            label += "0"
        converted_label = "".join([f"<|{char}|>" for char in label])
        return label_tokens[0] + converted_label + label_tokens[1]
    elif task in REACTION_BENCHMARKS + MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS:
        return label_tokens[0] + label + label_tokens[1]
    else:
        raise NotImplementedError


from torch_geometric.data import InMemoryDataset, Data


# TODO use task or refactor it
# getitem shoul return enough information that what is label, and what is the label meaning (to format instruction)
class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(
        self, data, task_subtask_pair, subtask_idx=0, prompt=None, debug=False
    ):
        self.debug = debug
        self.data = data
        self.subtask_idx = subtask_idx
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")
        self.prompt = prompt
        self.mol_special_tokens = self.prompt.replace(".", "").split("{}")

        if self.task in CLASSIFICATION_BENCHMARKS:
            if self.task in ["clintox"]:
                self.instruction_list = getattr(
                    instructions, f"{self.task}_{self.subtask}"
                )
            else:
                self.instruction_list = getattr(instructions, self.task)
            self.label_tokens = BOOL_TOKENS
        elif self.task in REGRESSION_BENCHMARKS:
            self.instruction_list = getattr(instructions, self.task)
            self.label_tokens = FLOAT_TOKENS
        else:
            raise NotImplementedError

        self.set_necessary_data()

    def __len__(self):
        return len(self.smiles_list)

    def get_necessary_data(self, index):
        smiles = self.smiles_list[index]
        label = self.label_list[index]
        label = wrap_label(label, self.task, self.mol_special_tokens)
        graph = smiles2data(smiles)
        # randomly select one instruction from list
        instruction = self.instruction_list[
            np.random.choice(len(self.instruction_list))
        ]

        if self.prompt.find("{}") >= 0:
            smiles_prompt = self.prompt.format(smiles)
        else:
            smiles_prompt = self.prompt

        return graph, label, smiles_prompt, instruction

    def set_necessary_data(self):
        self.mol_list = self.data.X
        self.label_list = self.data.y[:, self.subtask_idx]
        if self.task in REGRESSION_BENCHMARKS:
            self.label_stat = {
                "avg": np.array(self.label_list).mean(),
                "std": np.array(self.label_list).std(),
                "min": np.array(self.label_list).min(),
                "max": np.array(self.label_list).max(),
            }
            # convert label into string
            self.label_list = [str(label) for label in self.label_list]
            self.label_stat.update(
                {
                    "avg_len": sum([len(label) for label in self.label_list])
                    / len(self.label_list),
                    "max_len": max([len(label) for label in self.label_list]),
                    "min_len": min([len(label) for label in self.label_list]),
                }
            )

        if self.debug:
            self.mol_list = self.mol_list[:100]
            self.label_list = self.label_list[:100]

        self.smiles_list = []
        for mol in self.mol_list:
            self.smiles_list.append(Chem.MolToSmiles(mol))

        label_list = []
        smiles_prompt_list = []
        graph_list = []
        instruction_list = []

        self.count_invalid_smiles = 0

        iter_bar = tqdm(
            range(len(self.mol_list)), total=len(self.mol_list), desc=self.task
        )
        for i in iter_bar:
            try:
                graph, label, smiles_prompt, instruction = self.get_necessary_data(i)
                label_list.append(label)
                smiles_prompt_list.append(smiles_prompt)
                graph_list.append(graph)
                instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.mol_list)}"
            )

        self.label_list = label_list
        self.smiles_prompt_list = smiles_prompt_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        smiles_prompt = self.smiles_prompt_list[index]
        instruction = self.instruction_list[index]

        return graph, label, smiles_prompt, self.task_subtask_pair, instruction


class MolInstructionDatset(Dataset):
    def __init__(self, data, task_subtask_pair, prompt=None, debug=False):
        self.debug = debug
        self.data = data
        self.prompt = prompt
        self.mol_special_tokens = self.prompt.replace(".", "").split("{}")
        # TODO: implement conversion to smiles or selfies controlled by this attribute
        # currently, only smiles is supported
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        self.set_necesary_data()

    def set_necesary_data(self):
        if self.debug:
            self.data = self.data[:100]
        else:
            self.data = self.data

        self.input_list = self.data["input"]
        self.label_list = self.data["output"]
        if self.task in REGRESSION_BENCHMARKS:
            self.label_list = [float(label) for label in self.label_list]

            self.label_stat = {
                "avg": np.array(self.label_list).mean(),
                "std": np.array(self.label_list).std(),
                "min": np.array(self.label_list).min(),
                "max": np.array(self.label_list).max(),
            }
            # convert label into string
            self.label_list = [str(label) for label in self.label_list]
            self.label_stat.update(
                {
                    "avg_len": sum([len(label) for label in self.label_list])
                    / len(self.label_list),
                    "max_len": max([len(label) for label in self.label_list]),
                    "min_len": min([len(label) for label in self.label_list]),
                }
            )

        self.instruction_list = self.data["instruction"]

        input_list = []
        label_list = []
        smiles_prompt_list = []
        graph_list = []
        instruction_list = []

        self.count_invalid_smiles = 0
        iter_bar = tqdm(
            range(len(self.input_list)), total=len(self.input_list), desc=self.task
        )
        for i in iter_bar:
            try:
                graph, label, smiles_prompt, instruction = self.get_necessary_data(i)
                input_list.append(self.input_list[i])
                label_list.append(label)
                smiles_prompt_list.append(smiles_prompt)
                graph_list.append(graph)
                instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.input_list)}"
            )

        self.input_list = input_list
        self.label_list = label_list
        self.smiles_prompt_list = smiles_prompt_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __len__(self):
        return len(self.input_list)

    def get_necessary_data(self, index):
        instruction = self.instruction_list[index]
        input = self.input_list[index]
        label = self.label_list[index]
        # one smiles in output
        if self.task in TEXT2MOL_BENCHMARKS:
            selfies = label  # label in mol-instruction dataset is annotated as selfies
            smiles = selfies.decoder(label)  # molca task smiles instead of selfies
            label = smiles
            # output smiles do not need to be converted to graph
            # but assign graph = None retrieve error in torch_geometric, so assign arbitral graph
            graph = smiles2data(
                "C"
            )  # output smiles do not need to be converted to graph
        elif self.task in REACTION_BENCHMARKS:
            # two smiles in input
            if self.task in ["reagent_prediction"]:
                two_selfies = input
                list_selfies = two_selfies.split(">>")
                list_smiles = [selfies.decoder(s) for s in list_selfies]
                smiles = (">>").join(list_smiles)
                graph = [smiles2data(s) for s in list_smiles]
            # one smiles in input and one smiles in output
            else:
                input_selfies = input
                smiles = selfies.decoder(input_selfies)
                graph = smiles2data(smiles)

            # TODO: reprocess training data. currently reaction data input smiles and output selfies
            output_selfies = label
            output_smiles = selfies.decoder(output_selfies)
            label = output_smiles

        else:
            # one smiles in input
            selfies = input
            smiles = selfies.decoder(selfies)
            graph = smiles2data(smiles)

        label = wrap_label(label, self.task, self.mol_special_tokens)

        if self.prompt.find("{}") >= 0:
            smiles_prompt = self.prompt.format(smiles)
        else:
            smiles_prompt = self.prompt

        return graph, label, smiles_prompt, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        smiles_prompt = self.smiles_prompt_list[index]
        instruction = self.instruction_list[index]

        return graph, label, smiles_prompt, self.task_subtask_pair, instruction


from ogb.utils import smiles2graph


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph["node_feat"])
    edge_index = torch.from_numpy(
        graph["edge_index"],
    )
    edge_attr = torch.from_numpy(graph["edge_feat"])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# Initialize with the data_list from ConcatDataset
class InstructionInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        filename,
        transform=None,
        pre_transform=None,
        prompt=None,
        resize=None,
        mol_string_conversion=False,
    ):
        self.filename = filename  # raw_file_names and processed_file_names use this
        self.prompt = prompt
        self.start, self.end = self.prompt.split("{}")
        self.resize = resize
        self.mol_string_conversion = mol_string_conversion
        print(f"Convert mol: {self.mol_string_conversion}")
        super(InstructionInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        if self.resize:
            self.shuffle_dataset()
            self.reduce_dataset_size(self.resize)

    def shuffle_dataset(self):
        # Shuffle the dataset
        data_list = [self.get(i) for i in range(len(self))]
        import random

        random.shuffle(data_list)
        data, slices = self.collate(data_list)
        self.data = data
        self.slices = slices

    def reduce_dataset_size(self, new_size):
        # Check if new size is smaller than the current size
        current_size = list(self.slices.values())[0].size(0) - 1
        if new_size >= current_size:
            print("New size must be smaller than the current dataset size.")
            return

        # Adjust data
        for key in self.slices.keys():
            self.slices[key] = self.slices[key][: new_size + 1]

        # Slice the data according to new slices
        reduced_data = {}
        for key, item in self.data:
            start = self.slices[key][0].item()
            end = self.slices[key][-1].item()
            reduced_data[key] = item[start:end]

        self.data = Data(**reduced_data)
        print(f"Dataset size reduced to {new_size}")

    @property
    def raw_file_names(self):
        return f"{self.filename}.pth"

    @property
    def processed_file_names(self):
        return f"{self.filename}.pt"

    def download(self):
        def get_dataset(task_name):
            base_path = f"dataset/{task_name}"
            os.makedirs(base_path, exist_ok=True)

            # get dataset from deepchem
            if task_name == "bace":
                loading_fn = dc.molnet.load_bace_classification
            elif task_name == "esol":
                loading_fn = dc.molnet.load_delaney
            elif task_name in REGRESSION_BENCHMARKS and "qm9" not in task_name:
                loading_fn = getattr(dc.molnet, f"load_{task_name}")
            elif task_name in CLASSIFICATION_BENCHMARKS:
                loading_fn = getattr(dc.molnet, f"load_{task_name}")

            # mol-instruction datasets
            elif (
                task_name
                in MOL2TEXT_BENCHMARKS
                + TEXT2MOL_BENCHMARKS
                + REACTION_BENCHMARKS
                + ["qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap"]
            ):
                mol_instruction_dataset = load_dataset(
                    "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
                )
                if "qm9" in task_name:
                    dataset = mol_instruction_dataset["property_prediction"]
                    subtask_name = task_name.split("_")[1]
                    subtask_instructions = getattr(instructions, subtask_name)
                    dataset = dataset.filter(
                        lambda x: x["instruction"] in subtask_instructions
                    )
                else:
                    dataset = mol_instruction_dataset[task_name]

                train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
                split = train_dataset.train_test_split(test_size=0.02, shuffle=True)
                train_dataset, valid_dataset = split["train"], split["test"]

                test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
                tasks = [task_name]
            else:
                raise NotImplementedError

            # dataset from deepchem
            if (
                task_name in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS
                and "qm9" not in task_name
            ):
                tasks, datasets, transformers = loading_fn(
                    featurizer="Raw",
                    splitter="scaffold",
                    save_dir=base_path,
                    data_dir=base_path,
                    reload=True,
                )
                train_dataset, valid_dataset, test_dataset = datasets
            else:
                # dataset from mol-instruction is already loaded
                pass

            return tasks, train_dataset, valid_dataset, test_dataset

        # subtask index is necessary when loading from deepchem
        task_subtask_lists = {
            "qm9_homo": [0],
            "qm9_lumo": [0],
            "qm9_homo_lumo_gap": [0],
            "reagent_prediction": [0],
            "forward_reaction_prediction": [0],
            "retrosynthesis": [0],
            "bace": [0],
            "bbbp": [0],
            "clintox": [0, 1],
            "toxcast": [0],
            "sider": [0],
            "tox21": [0],
            "hiv": [0],
            "esol": [0],
            "lipo": [0],
            "description_guided_molecule_design": [0],
            "molecular_description_generation": [0],
        }

        target_benchmarks = []
        if "classification" in self.filename:
            target_benchmarks = CLASSIFICATION_BENCHMARKS
        elif "regression" in self.filename:
            target_benchmarks = REGRESSION_BENCHMARKS
        elif "reaction" in self.filename:
            target_benchmarks = REACTION_BENCHMARKS
        elif "translation" in self.filename:
            target_benchmarks = MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS

        # leave task only if it is in target_benchmarks
        task_subtask_pairs = [
            (task, subtask) if task in target_benchmarks else None
            for task, subtasks in task_subtask_lists.items()
            for subtask in subtasks
        ]

        # remove None
        self.task_subtask_pairs = [t for t in task_subtask_pairs if t]

        multi_task_datasets = {
            task_name: get_dataset(
                task_name=task_name
            )  # {task_name: [train, val, test]
            for task_name in target_benchmarks
        }

        train_datasets, val_datasets, test_datasets = [], [], []
        for task_subtask_pair in tqdm(
            self.task_subtask_pairs, desc="Processing task_subtask_pairs"
        ):
            task_name = task_subtask_pair[0]
            subtasks = multi_task_datasets[task_name][0]
            subtask_idx = task_subtask_pair[1]
            task_subtask_pair = f"{task_name}/{subtasks[subtask_idx]}"

            data_split = multi_task_datasets[task_name][
                1:
            ]  # train_set, val_set, test_set
            # dataset processed via MoleculeNetDatasetDeepChem
            if (
                task_name in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS
                and "qm9" not in task_name
            ):
                train_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
                valid_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
                test_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
            # qm9 in regression benchmark is processed via MolInstructionDataset
            elif (
                task_name
                in TEXT2MOL_BENCHMARKS
                + MOL2TEXT_BENCHMARKS
                + REACTION_BENCHMARKS
                + REGRESSION_BENCHMARKS
            ):
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    debug=self.debug,
                )

            train_datasets.append(train_dataset)
            val_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)

        self.task = self.filename.split("_")[0]
        # save 3 split at the same time, so to skip redundant processing for validation and test set
        concat_datasets = {"train": [], "val": [], "test": []}

        for i in range(len(self.task_subtask_pairs)):
            task_subtask_pair = self.task_subtask_pairs[i]
            task_name = task_subtask_pair[0]

            concat_datasets["train"].append(train_datasets[i])
            concat_datasets["val"].append(val_datasets[i])
            concat_datasets["test"].append(test_datasets[i])

        for split in ["train", "val", "test"]:
            concat_dataset = ConcatDataset(concat_datasets[split])
            torch.save(
                concat_dataset,
                f"{self.raw_dir}/{self.task}_{split}.pth",
            )

        print("Saved dataset for task: ", self.task)

    # instance = [Data, label, smiles_prompt, task_subtask_pair, instruction]
    def convert_mol_string(self, instance):
        if self.mol_string_conversion == "smiles2selfies":
            convert = selfies.encoder
        elif self.mol_string_conversion == "selfies2smiles":
            convert = selfies.decoder
        else:
            return instance  # no conversion

        # start string representation conversion
        task = instance[3].split("/")[0]

        # convert output
        if (
            task
            in MOL2TEXT_BENCHMARKS + REGRESSION_BENCHMARKS + CLASSIFICATION_BENCHMARKS
        ):
            label = instance[1]
        elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
            mol_label = instance[1].split(self.start)[1].split(self.end)[0]
            label = convert(mol_label)
        else:
            raise NotImplementedError

        # convert input
        if task in "reagent_prediction":
            mol_strings = (
                instance[2].split(self.start)[1].split(self.end)[0].split(">>")
            )
            input_mol_string = ">>".join([convert(mol) for mol in mol_strings])
        elif task in TEXT2MOL_BENCHMARKS:
            input_mol_string = instance[2]
        elif (
            task
            in REACTION_BENCHMARKS
            + CLASSIFICATION_BENCHMARKS
            + REGRESSION_BENCHMARKS
            + MOL2TEXT_BENCHMARKS
        ):
            mol_string = instance[2].split(self.start)[1].split(self.end)[0]
            input_mol_string = convert(mol_string)
        else:
            raise NotImplementedError

        converted_instance = (
            instance[0],
            label,
            input_mol_string,
            instance[3],
            instance[4],
        )

        return converted_instance

    def process(self):
        # Process data_list and store in `self.data` and `self.slices`
        raw_data_list = list(torch.load(self.raw_paths[0]))
        data_list = []
        count_fail_conversion = 0

        iter_bar = tqdm(range(len(raw_data_list)))
        for i in iter_bar:
            iter_bar.set_description(
                f"{self.filename}|Num fail: {count_fail_conversion}|Ratio fail: {count_fail_conversion/(i+1)}"
            )
            instance = raw_data_list[i]
            try:
                instance = self.convert_mol_string(instance)

                if isinstance(instance[0], list):
                    # reagent prediction dataset
                    # input string: reactant>>product / output string: reagent
                    data = Data(
                        x=instance[0][0].x,
                        edge_index=instance[0][0].edge_index,
                        edge_attr=instance[0][0].edge_attr,
                        y=instance[1],
                        smiles_prompt=instance[2],
                        task_subtask_pair=instance[3],
                        instruction=instance[4],
                    )
                    # ~_{i} are for additional molecules, such as reagent prediction
                    """
                    # TODO: temporally remove, just to run string_only learning now.
                    # later, we need to implement the multi-graph learning with proper manner
                    data.__setattr__(f"x_1", instance[0][1])
                    data.__setattr__(f"edge_index_1", instance[0][1].edge_index)
                    data.__setattr__(f"edge_attr_1", instance[0][1].edge_attr)
                    """
                else:
                    data = Data(
                        x=instance[0].x,
                        edge_index=instance[0].edge_index,
                        edge_attr=instance[0].edge_attr,
                        y=instance[1],
                        smiles_prompt=instance[2],
                        task_subtask_pair=instance[3],
                        instruction=instance[4],
                    )
                    # save dummy data for additional molecules, due to collate method sanity
                    if "reaction" in self.filename:
                        pass
                        """
                        data.__setattr__("x_1", instance[0].x)
                        data.__setattr__("edge_index_1", instance[0].edge_index)
                        data.__setattr__("edge_attr_1", instance[0].edge_attr)
                        """

                data_list.append(data)
            except:
                count_fail_conversion += 1
                continue

        self.save(data_list, self.processed_paths[0])
        print("Saved processed dataset for task: ", self.filename)

    def __getitem__(self, index):
        data = self.get(index)
        label = data.y
        smiles_prompt = data.smiles_prompt
        task_subtask_pair = data.task_subtask_pair
        instruction = data.instruction

        return data, label, smiles_prompt, task_subtask_pair, instruction


if __name__ == "__main__":
    dm = Stage3DM(
        mode="pretrain",
        num_workers=0,
        batch_size=256,
        root="data/",
        text_max_len=128,
        tokenizer=None,
    )
