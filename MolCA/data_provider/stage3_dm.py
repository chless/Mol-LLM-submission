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
import selfies as sf
from tqdm import tqdm
import model.added_tokens as added_tokens
import random

# we split individual characters inside special tokens like [START_DNA]
# TODO: change this ugly I_SMILES things to regular special token, and add the special token to vocab whichever LLM
# CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")
CUSTOM_SEQ_RE = re.compile(r"(<MOL_1D>)(.*?)(</MOL_1D>)")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.


def prepare_llm_input(
    mol_string,
    instruction,
    task,
    mol_ph,
    mol_representation,
):
    if CUSTOM_SEQ_RE.match(mol_string) is None:
        mol_string = added_tokens.MOL_1D[0] + mol_string + added_tokens.MOL_1D[1]

    mol_ph = added_tokens.MOL_2D[0] + mol_ph + added_tokens.MOL_2D[1]

    if mol_representation == "graph_only":
        mol_string_converted = CUSTOM_SEQ_RE.sub(r"%s" % (mol_ph), mol_string)
        # reagent prediction has reaction direction token and second molecule

    elif mol_representation == "string_only":
        mol_string_converted = CUSTOM_SEQ_RE.sub(r"\1\2\3", mol_string)

    elif mol_representation == "string+graph":
        mol_string_converted = CUSTOM_SEQ_RE.sub(r"\1\2\3%s" % (mol_ph), mol_string)
    else:
        raise NotImplementedError("mol_representation should be one of the options")

    # for tasks whose input does not contain molecule string (such as text2mol), don't add mol_string
    if not "<None>" in mol_string:
        llm_prompt = instruction + mol_string_converted
    else:
        llm_prompt = instruction

    llm_prompt += " Answer format: {label_start}...{label_end}."
    if task in CLASSIFICATION_BENCHMARKS:
        llm_prompt = llm_prompt.replace("{label_start}", added_tokens.BOOL[0])
        llm_prompt = llm_prompt.replace("{label_end}", added_tokens.BOOL[1])
    elif task in REGRESSION_BENCHMARKS:
        llm_prompt = llm_prompt.replace("{label_start}", added_tokens.FLOAT[0])
        llm_prompt = llm_prompt.replace("{label_end}", added_tokens.FLOAT[1])
    elif task in MOL2TEXT_BENCHMARKS:
        llm_prompt = llm_prompt.replace("{label_start}", added_tokens.DESCRIPTION[0])
        llm_prompt = llm_prompt.replace("{label_end}", added_tokens.DESCRIPTION[1])
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        try:
            llm_prompt = llm_prompt.replace("{label_start}", added_tokens.MOL_1D[0])
            llm_prompt = llm_prompt.replace("{label_end}", added_tokens.MOL_1D[1])
        except:
            print(llm_prompt)
            raise NotImplementedError
    else:
        raise NotImplementedError

    return llm_prompt


class DataCollater:
    def __init__(
        self,
        tokenizer,
        prompt_max_len,
        label_max_len,
        mol_ph,
        mol_token_id,
        mol_representation=True,
        model=None,
        truncation=True,
        padding="max_length",
        fit_llm_input_convention=None,
        fit_llm_output_convention=None,
    ):
        self.prompt_max_len = prompt_max_len
        self.label_max_len = label_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation
        self.model = model
        self.truncation = bool(truncation)
        self.padding = padding
        self.fit_llm_input_convention = fit_llm_input_convention
        self.fit_llm_output_convention = fit_llm_output_convention

    def __call__(self, batch):
        graphs, label_texts, input_mol_string, tasks, instructions = zip(*batch)

        if isinstance(graphs[0], PairData):
            reactant_batch = torch.tensor([], dtype=torch.int64)
            product_batch = torch.tensor([], dtype=torch.int64)
            for i in range(len(graphs)):
                reactant_num_nodes = graphs[i].reactant_x.size(0)
                reactant_node_indexing_tensor = torch.tensor(
                    [i] * reactant_num_nodes, dtype=torch.int64
                )
                reactant_batch = torch.cat(
                    (reactant_batch, reactant_node_indexing_tensor), 0
                )
                product_num_nodes = graphs[i].product_x.size(0)
                product_node_indexing_tensor = torch.tensor(
                    [i] * product_num_nodes, dtype=torch.int64
                )
                product_batch = torch.cat(
                    (product_batch, product_node_indexing_tensor), 0
                )

        graphs = self.collater(graphs)
        if isinstance(graphs, PairData):
            graphs.reactant_batch = reactant_batch
            graphs.product_batch = product_batch

        ## deal with prompt
        input_texts = [
            prepare_llm_input(
                mol_string=mol_string,
                instruction=instruction,
                task=task.split("/")[0],
                mol_ph=self.mol_ph,
                mol_representation=self.mol_representation,
            )
            for mol_string, instruction, task in zip(
                input_mol_string, instructions, tasks
            )
        ]
        input_texts = [self.fit_llm_input_convention(text) for text in input_texts]

        self.tokenizer.padding_side = "left"
        input_tokens = self.tokenizer(
            text=input_texts,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=False,
            max_length=self.prompt_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        is_mol_token = input_tokens.input_ids == self.mol_token_id
        input_tokens["is_mol_token"] = is_mol_token

        # concat eos token to the end of the label
        label_texts = [self.fit_llm_output_convention(label) for label in label_texts]

        self.tokenizer.padding_side = "right"
        label_tokens = self.tokenizer(
            text=label_texts,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=False,
            max_length=self.label_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return graphs, input_tokens, label_tokens, tasks


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

MOL2TEXT_BENCHMARKS = [
    # "molecular_description_generation",
    "chebi-20-mol2text"
]

TEXT2MOL_BENCHMARKS = [
    # "description_guided_molecule_design",
    "chebi-20-text2mol"
]

REACTION_BENCHMARKS = [
    "reagent_prediction",
    "forward_reaction_prediction",
    "retrosynthesis",
]

TOTAL_BENCHMARKS = (
    REACTION_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
)


class Stage3DM(LightningDataModule):
    def __init__(
        self,
        mode: str = "pretrain",
        num_workers: int = 0,
        root: str = "data/",
        tokenizer=None,
        fit_llm_input_convention=None,
        fit_llm_output_convention=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.num_workers = num_workers
        self.prompt_max_len = args.prompt_max_len
        self.label_max_len = args.label_max_len
        self.debug = args.debug
        self.args = args
        self.root = root
        self.fit_llm_input_convention = fit_llm_input_convention
        self.fit_llm_output_convention = fit_llm_output_convention

        self.batch_sizes = {
            "classification": args.per_device_batch_size_cls,
            "regression": args.per_device_batch_size_reg,
            "reaction": args.per_device_batch_size_rxn,
            "reagent": args.per_device_batch_size_rea,
            "translation": args.per_device_batch_size_trn,
        }
        self.inference_batch_sizes = {
            "classification": args.per_device_inference_batch_size_cls,
            "regression": args.per_device_inference_batch_size_reg,
            "reaction": args.per_device_inference_batch_size_rxn,
            "reagent": args.per_device_inference_batch_size_rea,
            "translation": args.per_device_inference_batch_size_trn,
        }
        self.label_max_lens = {
            "classification": 6,
            "regression": 12,
            "reaction": args.label_max_len,
            "reagent": args.label_max_len,
            "translation": args.label_max_len,
        }
        self.prompt_max_lens = {
            "classification": args.prompt_max_len - 6,
            "regression": args.prompt_max_len - 12,
            "reaction": args.prompt_max_len,
            "reagent": args.prompt_max_len,
            "translation": args.prompt_max_len,
        }

        if root == "multi_task":
            self.task_categories = [
                "translation",
                "reagent",
                "reaction",
                "regression",
                "classification",
            ]
        elif root in self.task_categories:
            self.task_categories = [root]
        elif root == "smol_instruct":
            self.task_categories = ["smol_instruct"]
        else:
            raise NotImplementedError
        
        self.concat_datasets = {
            task: {"train": None, "val": None, "test": None}
            for task in self.task_categories
        }
        for task in self.concat_datasets.keys():
            for split in ["train", "val", "test"]:
                if split == "val":
                    resize = args.valset_resize if args.valset_resize > 0 else None
                elif split == "test":
                    resize = args.testset_resize if args.testset_resize > 0 else None
                elif split == "train":
                    resize = args.trainset_resize if args.trainset_resize > 0 else None
                else:
                    raise NotImplementedError

                self.concat_datasets[task][split] = InstructionInMemoryDataset(
                    root=self.args.raw_data_root,
                    filename=f"{task}_{split}",
                    resize=resize,
                )

        self.init_tokenizer(tokenizer)
        self.mol_ph_token = "<mol>" * self.args.num_query_token
        self.mol_representation = args.mol_representation

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # self.train_dataset.tokenizer = tokenizer
        # self.val_dataset.tokenizer = tokenizer
        # self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        loader = []
        for task in self.concat_datasets.keys():
            loader.append(
                DataLoader(
                    self.concat_datasets[task]["train"],
                    batch_size=self.batch_sizes[task],
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=True,
                    collate_fn=DataCollater(
                        tokenizer=self.tokenizer,
                        prompt_max_len=self.prompt_max_lens[task],
                        label_max_len=self.label_max_lens[task],
                        mol_ph=self.mol_ph_token,
                        mol_token_id=self.mol_token_id,
                        mol_representation=self.mol_representation,
                        model=self.args.llm_model,
                        truncation=self.args.truncation,
                        padding=self.args.padding,
                        fit_llm_input_convention=self.fit_llm_input_convention,
                        fit_llm_output_convention=self.fit_llm_output_convention,
                    ),
                )
            )
        return loader

    def val_dataloader(self):
        loader = []
        for task in self.concat_datasets.keys():
            if task in ["classification", "regression"]:
                label_max_len = 9
                prompt_max_len = self.prompt_max_len - 9
            elif task in ["reagent"]:
                label_max_len = self.label_max_len
                prompt_max_len = self.prompt_max_len + self.args.num_query_token
            else:
                label_max_len = self.label_max_len
                prompt_max_len = self.prompt_max_len

            loader.append(
                DataLoader(
                    self.concat_datasets[task]["val"],
                    batch_size=self.inference_batch_sizes[task],
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=DataCollater(
                        tokenizer=self.tokenizer,
                        prompt_max_len=prompt_max_len,
                        label_max_len=label_max_len,
                        mol_ph=self.mol_ph_token,
                        mol_token_id=self.mol_token_id,
                        mol_representation=self.mol_representation,
                        model=self.args.llm_model,
                        truncation=self.args.truncation,
                        padding=self.args.padding,
                        fit_llm_input_convention=self.fit_llm_input_convention,
                        fit_llm_output_convention=self.fit_llm_output_convention,
                    ),
                )
            )
        return loader

    def test_dataloader(self):
        loader = []
        split = "test" if not self.args.test_on_trainset else "train"
        for task in self.concat_datasets.keys():
            if task in ["classification", "regression"]:
                label_max_len = 9
                prompt_max_len = self.prompt_max_len - 9
            elif task in ["reagent"]:
                label_max_len = self.label_max_len
                prompt_max_len = self.prompt_max_len + self.args.num_query_token
            else:
                label_max_len = self.label_max_len
                prompt_max_len = self.prompt_max_len

            loader.append(
                DataLoader(
                    self.concat_datasets[task][split],
                    batch_size=self.inference_batch_sizes[task],
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=DataCollater(
                        tokenizer=self.tokenizer,
                        prompt_max_len=prompt_max_len,
                        label_max_len=label_max_len,
                        mol_ph=self.mol_ph_token,
                        mol_token_id=self.mol_token_id,
                        mol_representation=self.mol_representation,
                        model=self.args.llm_model,
                        truncation=self.args.truncation,
                        padding=self.args.padding,
                        fit_llm_input_convention=self.fit_llm_input_convention,
                        fit_llm_output_convention=self.fit_llm_output_convention,
                    ),
                )
            )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--per_device_batch_size_cls", type=int, default=32)
        parser.add_argument("--per_device_batch_size_reg", type=int, default=32)
        parser.add_argument("--per_device_batch_size_rxn", type=int, default=32)
        parser.add_argument("--per_device_batch_size_rea", type=int, default=32)
        parser.add_argument("--per_device_batch_size_trn", type=int, default=32)
        parser.add_argument(
            "--per_device_inference_batch_size_cls", type=int, default=4
        )
        parser.add_argument(
            "--per_device_inference_batch_size_reg", type=int, default=4
        )
        parser.add_argument(
            "--per_device_inference_batch_size_rxn", type=int, default=4
        )
        parser.add_argument(
            "--per_device_inference_batch_size_rea", type=int, default=4
        )
        parser.add_argument(
            "--per_device_inference_batch_size_trn", type=int, default=4
        )
        parser.add_argument("--use_smiles", action="store_true", default=False)
        parser.add_argument("--root", type=str, default="data/PubChemDataset_v4")
        parser.add_argument("--prompt_max_len", type=int, default=512)
        parser.add_argument("--label_max_len", type=int, default=256)
        parser.add_argument("--truncation", default=1, type=int)
        parser.add_argument("--padding", default="max_length", type=str)
        parser.add_argument(
            "--prompt",
            type=str,
            default="[START_I_SMILES]{}[END_I_SMILES]",
        )  # not used at all in stage 3
        parser.add_argument("--filtered_cid_path", type=str, default=None)

        # moleculenet dataset
        parser.add_argument("--subtask_idx", type=int, default=0)
        parser.add_argument(
            "--raw_data_root", type=str, default="MolCA/data/multi_task_dataset"
        )
        parser.add_argument("--mol_string_randomization_ratio", type=float, default=-1)

        return parent_parser


from tqdm import tqdm

from rdkit import Chem


def wrap_label(label, task):

    if task in CLASSIFICATION_BENCHMARKS:
        label_tokens = added_tokens.BOOL
    elif task in REGRESSION_BENCHMARKS:
        label_tokens = added_tokens.FLOAT
    elif task in MOL2TEXT_BENCHMARKS:
        label_tokens = added_tokens.DESCRIPTION
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        label_tokens = added_tokens.MOL_1D
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

        if self.task in CLASSIFICATION_BENCHMARKS:
            if self.task in ["clintox"]:
                self.instruction_list = getattr(
                    instructions, f"{self.task}_{self.subtask}"
                )
            else:
                self.instruction_list = getattr(instructions, self.task)
            self.label_tokens = added_tokens.BOOL
        elif self.task in REGRESSION_BENCHMARKS:
            self.instruction_list = getattr(instructions, self.task)
            self.label_tokens = added_tokens.FLOAT
        else:
            raise NotImplementedError

        self.set_necessary_data()

    def __len__(self):
        return len(self.smiles_list)

    def get_necessary_data(self, index):
        smiles = self.smiles_list[index]
        # set molecule string representation as selfies
        input_mol_string = sf.encoder(smiles)
        input_mol_string = (
            added_tokens.MOL_1D[0] + input_mol_string + added_tokens.MOL_1D[1]
        )
        label = self.label_list[index]
        label = wrap_label(label, self.task)
        graph = smiles2data(smiles)
        # randomly select one instruction from list
        instruction = self.instruction_list[
            np.random.choice(len(self.instruction_list))
        ]
        return graph, label, input_mol_string, instruction

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
        input_mol_string_list = []
        graph_list = []
        instruction_list = []

        self.count_invalid_smiles = 0

        iter_bar = tqdm(
            range(len(self.mol_list)), total=len(self.mol_list), desc=self.task
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                label_list.append(label)
                input_mol_string_list.append(input_mol_string)
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
        self.input_mol_string_list = input_mol_string_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


class MolInstructionDatset(Dataset):
    def __init__(self, data, task_subtask_pair, prompt=None, debug=False):
        self.debug = debug
        self.data = data
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
        input_mol_string_list = []
        graph_list = []
        instruction_list = []

        self.count_invalid_smiles = 0
        iter_bar = tqdm(
            range(len(self.input_list)), total=len(self.input_list), desc=self.task
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                input_list.append(self.input_list[i])
                label_list.append(label)
                input_mol_string_list.append(input_mol_string)
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
        self.input_mol_string_list = input_mol_string_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __len__(self):
        return len(self.input_list)

    def get_necessary_data(self, index):
        instruction = self.instruction_list[index]
        input = self.input_list[index]  # if mol_string, representation is selfies
        label = self.label_list[index]  # if mol_string, representation is selfies
        # one smiles in output
        if self.task in TEXT2MOL_BENCHMARKS:
            # output smiles do not need to be converted to graph
            # but assign graph = None retrieve error in torch_geometric, so assign graph label intended as null graph
            instruction += "\n" + input
            graph = smiles2data(
                sf.decoder(label)
            )  # output smiles do not need to be converted to graph
            input_mol_string = "<None>"  # no input molstring in text2mol
        elif self.task in REACTION_BENCHMARKS:
            # two smiles in input0
            if self.task in ["reagent_prediction"]:
                assert ">>" in input
                list_selfies = input.split(
                    ">>"
                )  # reagent prediction has two selfies in input
                input_mol_string = input.replace(
                    ">>",
                    f"{added_tokens.MOL_1D[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.MOL_1D[0]}",
                )
                list_smiles = [sf.decoder(s) for s in list_selfies]
                graph = [smiles2data(s) for s in list_smiles]
            # one smiles in input and one smiles in output
            else:
                input_mol_string = input
                smiles = sf.decoder(input_mol_string)
                graph = smiles2data(smiles)

        else:
            # one selfies in input
            input_mol_string = input
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)

        label = wrap_label(label, self.task)
        input_mol_string = (
            added_tokens.MOL_1D[0] + input_mol_string + added_tokens.MOL_1D[1]
        )

        return graph, label, input_mol_string, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


class ChEBIDatset(Dataset):
    def __init__(self, data, task_subtask_pair, prompt=None, debug=False):
        self.debug = debug
        self.data = data
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

        self.description_list = self.data["description"]
        self.selfies_list = self.data["SELFIES"]
        self.smiles_list = self.data["SMILES"]
        self.instruction_list = getattr(instructions, self.task.replace("-", "_"))

        input_mol_string_list = []
        graph_list = []
        instruction_list = []
        label_list = []

        self.count_invalid_smiles = 0
        iter_bar = tqdm(
            range(len(self.description_list)),
            total=len(self.description_list),
            desc=self.task,
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                label_list.append(label)
                input_mol_string_list.append(input_mol_string)
                graph_list.append(graph)
                instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.label_list)}"
            )

        self.label_list = label_list
        self.input_mol_string_list = input_mol_string_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index):
        instruction = self.instruction_list[
            np.random.choice(len(self.instruction_list))
        ]
        descriptiopn = self.description_list[index]
        selfies = self.selfies_list[index]
        smiles = self.smiles_list[index]

        if self.task in TEXT2MOL_BENCHMARKS:
            label = selfies
            instruction += (
                "\n"
                + added_tokens.DESCRIPTION[0]
                + descriptiopn
                + added_tokens.DESCRIPTION[1]
            )
            graph = smiles2data(smiles)
            input_mol_string = "<None>"
        elif self.task in MOL2TEXT_BENCHMARKS:
            label = descriptiopn
            input_mol_string = selfies
            graph = smiles2data(smiles)

        label = wrap_label(label, self.task)
        input_mol_string = (
            added_tokens.MOL_1D[0] + input_mol_string + added_tokens.MOL_1D[1]
        )

        return graph, label, input_mol_string, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


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


# torch_geometric.data.Data variants for paired graph data, i.e. reagent prediction
class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "reactant_edge_index":
            return self.reactant_x.size(0)
        if key == "product_edge_index":
            return self.product_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


# Initialize with the data_list from ConcatDataset
class InstructionInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        filename,
        transform=None,
        pre_transform=None,
        resize=None,
    ):
        self.filename = filename  # raw_file_names and processed_file_names use this
        self.start, self.end = added_tokens.MOL_1D
        self.resize = resize
        super(InstructionInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        if self.resize:
            self.shuffle_dataset()
            self.reduce_dataset_size(self.resize)

    def shuffle_dataset(self):
        # Shuffle the dataset
        data_list = [self.get(i) for i in range(len(self))]

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

        # though most datasets use Data class, some datasets use PairData class
        # the miss instantiation of PairData class results in error when collate, due to malfunctioning of __inc__
        data_class = type(self._data)
        self.data = data_class(**reduced_data)
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
            elif "chebi-20" in task_name:
                dataset = load_dataset("liupf/ChEBI-20-MM")
                train_dataset = dataset["train"]
                valid_dataset = dataset["validation"]
                test_dataset = dataset["test"]
                tasks = [task_name]

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
            # "description_guided_molecule_design": [0],
            # "molecular_description_generation": [0],
            "chebi-20-mol2text": [0],
            "chebi-20-text2mol": [0],
        }

        target_benchmarks = []
        if "classification" in self.filename:
            target_benchmarks = CLASSIFICATION_BENCHMARKS
        elif "regression" in self.filename:
            target_benchmarks = REGRESSION_BENCHMARKS
        elif "reaction" in self.filename:
            target_benchmarks = [
                "forward_reaction_prediction",
                "retrosynthesis",
            ]
        elif "reagent" in self.filename:
            target_benchmarks = ["reagent_prediction"]
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
                    subtask_idx=subtask_idx,
                )
                valid_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    subtask_idx=subtask_idx,
                )
                test_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    subtask_idx=subtask_idx,
                )
            elif task_name in ["chebi-20-mol2text", "chebi-20-text2mol"]:
                train_dataset = ChEBIDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                )
                valid_dataset = ChEBIDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                )
                test_dataset = ChEBIDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
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
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
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
                if isinstance(instance[0], list):
                    # reagent prediction dataset
                    # input string: reactant>>product / output string: reagent
                    data = PairData(
                        reactant_x=instance[0][0].x,
                        reactant_edge_index=instance[0][0].edge_index,
                        reactant_edge_attr=instance[0][0].edge_attr,
                        product_x=instance[0][1].x,
                        product_edge_index=instance[0][1].edge_index,
                        product_edge_attr=instance[0][1].edge_attr,
                        y=instance[1],
                        input_mol_string=instance[2],
                        task_subtask_pair=instance[3],
                        instruction=instance[4],
                    )
                else:
                    data = Data(
                        x=instance[0].x,
                        edge_index=instance[0].edge_index,
                        edge_attr=instance[0].edge_attr,
                        y=instance[1],
                        input_mol_string=instance[2],
                        task_subtask_pair=instance[3],
                        instruction=instance[4],
                    )
                data_list.append(data)
            except:
                count_fail_conversion += 1
                continue

        self.save(data_list, self.processed_paths[0])
        print("Saved processed dataset for task: ", self.filename)

    def __getitem__(self, index):
        data = self.get(index)
        label = data.y
        if hasattr(data, "smiles_prompt"):
            input_mol_string = data.smiles_prompt
        else:
            input_mol_string = data.input_mol_string
        task_subtask_pair = data.task_subtask_pair
        instruction = data.instruction

        return data, label, input_mol_string, task_subtask_pair, instruction


if __name__ == "__main__":
    dm = Stage3DM(
        mode="pretrain",
        num_workers=0,
        batch_size=256,
        root="data/",
        text_max_len=128,
        tokenizer=None,
    )
