# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule

# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader.dataloader import Collater
import re
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import InMemoryDataset, Data
import os

import deepchem as dc
from datasets import load_dataset
from data_provider import instructions
import numpy as np
import selfies as sf
from tqdm import tqdm
import model.added_tokens as added_tokens
import random
from typing import Any


# we split individual characters inside special tokens like [START_DNA]
# TODO: change this ugly I_SMILES things to regular special token, and add the special token to vocab whichever LLM
# CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")
CUSTOM_SEQ_RE = re.compile(r"(<SELFIES>)(.*?)(</SELFIES>)")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.


def prepare_llm_input(
    mol_string,
    instruction,
    mol_ph,
    mol_representation,
):
    if CUSTOM_SEQ_RE.match(mol_string) is None:
        mol_string = added_tokens.SELFIES[0] + mol_string + added_tokens.SELFIES[1]

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
    if (
        "<INPUT>" in instruction and not "<None>" in mol_string
    ):  # for LlaSMol whose input contains <INPUT>
        llm_prompt = instruction.replace("<INPUT>", mol_string_converted)
    elif not "<None>" in mol_string:
        llm_prompt = instruction + mol_string_converted
    else:
        llm_prompt = instruction

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
            additional_batch = torch.tensor([], dtype=torch.int64)
            for i in range(len(graphs)):
                additional_num_nodes = graphs[i].additional_x.size(0)
                additional_node_indexing_tensor = torch.tensor(
                    [i] * additional_num_nodes, dtype=torch.int64
                )
                additional_batch = torch.cat(
                    (additional_batch, additional_node_indexing_tensor), 0
                )

        graphs = self.collater(graphs)

        if isinstance(graphs, PairData):
            graphs.additional_batch = additional_batch

        ## deal with prompt
        input_texts = [
            prepare_llm_input(
                mol_string=mol_string,
                instruction=instruction,
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
        return graphs, input_tokens, label_tokens


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
    "chebi-20-mol2text",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
    "smol-molecule_captioning",
]

TEXT2MOL_BENCHMARKS = [
    # "description_guided_molecule_design",
    "chebi-20-text2mol",
    "smol-name_conversion-i2f",
    "smol-name_conversion-i2s",
    "smol-molecule_generation",
]

REACTION_BENCHMARKS = [
    "reagent_prediction",
    "forward_reaction_prediction",
    "retrosynthesis",
    "smol-forward_synthesis",
    "smol-retrosynthesis",
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
        tokenizer=None,
        fit_llm_input_convention=None,
        fit_llm_output_convention=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.num_workers = num_workers
        self.prompt_max_lens = args.prompt_max_lens
        self.label_max_lens = args.label_max_lens
        self.args = args
        self.fit_llm_input_convention = fit_llm_input_convention
        self.fit_llm_output_convention = fit_llm_output_convention

        self.batch_sizes = args.batch_sizes
        self.inference_batch_sizes = args.inference_batch_sizes
        self.label_max_lens = args.label_max_lens
        self.prompt_max_lens = args.prompt_max_lens

        self.task_categories = list(self.args.target_benchmarks.keys())

        self.concat_datasets = {
            task: {"train": None, "val": None, "test": None}
            for task in self.task_categories
        }
        for task in self.concat_datasets.keys():
            for split in ["test", "val", "train"]:
                if split == "val":
                    resize = args.valset_resize if args.valset_resize > 0 else None
                elif split == "test":
                    resize = args.testset_resize if args.testset_resize > 0 else None
                elif split == "train":
                    resize = args.trainset_resize if args.trainset_resize > 0 else None
                else:
                    raise NotImplementedError

                if split == "val":
                    split = "test"

                self.concat_datasets[task][split] = Mol_LLM_Dataset(
                    root=self.args.raw_data_root,
                    filename=f"{task}_{split}",
                    resize=resize,
                    args=self.args,
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

    def test_dataloader(self):
        loader = []
        split = "test" if not self.args.test_on_trainset else "train"
        for task in self.concat_datasets.keys():
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


from tqdm import tqdm

from rdkit import Chem


def wrap_label(label, task):

    if task in CLASSIFICATION_BENCHMARKS:
        label_tokens = added_tokens.BOOL
    elif task in REGRESSION_BENCHMARKS:
        label_tokens = added_tokens.FLOAT
    elif task in ["smol-name_conversion-s2f", "smol-name_conversion-i2f"]:
        label_tokens = added_tokens.MOLFORMULA
    elif task == "smol-name_conversion-s2i":
        label_tokens = added_tokens.IUPAC
    elif task in MOL2TEXT_BENCHMARKS:
        label_tokens = added_tokens.DESCRIPTION
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
        label_tokens = added_tokens.SELFIES
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
    def __init__(self, data, task_subtask_pair, subtask_idx=0, prompt=None):
        self.data = data
        self.subtask_idx = subtask_idx
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        if self.task in CLASSIFICATION_BENCHMARKS:
            if self.task in ["clintox"]:
                self.instruction_templates = getattr(
                    instructions, f"{self.task}_{self.subtask}"
                )
            else:
                self.instruction_templates = getattr(instructions, self.task)
            self.label_tokens = added_tokens.BOOL
        elif self.task in REGRESSION_BENCHMARKS:
            self.instruction_templates = getattr(instructions, self.task)
            self.label_tokens = added_tokens.FLOAT
        else:
            raise NotImplementedError

        self.set_necessary_data()

    def get_necessary_data(self, index):
        smiles = self.smiles_list[index]
        # set molecule string representation as selfies
        input_mol_string = sf.encoder(smiles)
        input_mol_string = (
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )
        label = self.label_list[index]
        label = wrap_label(label, self.task)
        graph = smiles2data(smiles)
        # randomly select one instruction from list
        instruction = self.instruction_templates[
            np.random.choice(len(self.instruction_templates))
        ]
        return graph, label, input_mol_string, instruction

    def set_necessary_data(self):
        self.mol_list = self.data.X
        self.label_list = self.data.y[:, self.subtask_idx]

        self.smiles_list = []
        for mol in self.mol_list:
            self.smiles_list.append(Chem.MolToSmiles(mol))

        self.label_list = []
        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []

        self.count_invalid_smiles = 0

        iter_bar = tqdm(
            range(len(self.mol_list)), total=len(self.mol_list), desc=self.task
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                self.label_list.append(label)
                self.input_mol_string_list.append(input_mol_string)
                self.graph_list.append(graph)
                self.instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.mol_list)}"
            )

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


class MolInstructionDatset(Dataset):
    def __init__(self, data, task_subtask_pair, prompt=None):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        self.set_necesary_data()

    def set_necesary_data(self):
        self.input_list = self.data["input"][:]
        self.label_list = self.data["output"][:]
        self.instruction_list = self.data["instruction"][:]

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
        return len(self.label_list)

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
                    f"{added_tokens.SELFIES[1]}{added_tokens.REACTION_DIRECTION[0]}{added_tokens.SELFIES[0]}",
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
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
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
    def __init__(self, data, task_subtask_pair, prompt=None):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        self.set_necesary_data()

    def set_necesary_data(self):
        self.description_list = self.data["description"][:]
        self.selfies_list = self.data["SELFIES"][:]
        self.smiles_list = self.data["SMILES"][:]
        self.instruction_templates = getattr(instructions, self.task.replace("-", "_"))

        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []
        self.label_list = []

        self.count_invalid_smiles = 0
        iter_bar = tqdm(
            range(len(self.description_list)),
            total=len(self.description_list),
            desc=self.task,
        )
        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(i)
                self.label_list.append(label)
                self.input_mol_string_list.append(input_mol_string)
                self.graph_list.append(graph)
                self.instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.label_list)}"
            )

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index):
        instruction = self.instruction_templates[
            np.random.choice(len(self.instruction_templates))
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
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )

        return graph, label, input_mol_string, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        input_mol_string = self.input_mol_string_list[index]
        instruction = self.instruction_list[index]

        return graph, label, input_mol_string, self.task_subtask_pair, instruction


class SMolInstructDataset(Dataset):
    def __init__(self, data, task_subtask_pair, prompt=None):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")
        self.instruction_templates = getattr(instructions, self.task.replace("-", "_"))
        self.set_necesary_data()

    def set_necesary_data(self):
        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []
        self.label_list = []

        # pre-load data
        raw_inputs = self.data["raw_input"][:]
        raw_outputs = self.data["raw_output"][:]

        iter_bar = tqdm(
            range(len(self.data)),
            total=len(self.data),
            desc=self.task,
        )
        """
        import multiprocessing as mp
        num_procs = 200
        with mp.Pool(num_procs) as pool:
            outputs = pool.starmap(
                self.get_necessary_data, 
                [(i, raw_inputs[i], raw_outputs[i]) for i in range(len(raw_inputs))])

        """
        outputs = []
        for i in iter_bar:
            outputs.append(self.get_necessary_data(i, raw_inputs[i], raw_outputs[i]))
        for o in outputs:
            if isinstance(o, Exception):
                print(o)
                continue
            self.input_mol_string_list.append(o["input_mol_string"])
            self.graph_list.append(o["graph"])
            self.instruction_list.append(o["instruction"])
            self.label_list.append(o["label"])

        print(
            f"{self.task}: Invalid smiles ratio: {1.0 - len(self.label_list)/len(self.data)}"
        )

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index, raw_input, raw_output):
        try:
            raw_input = raw_input
            label = raw_output
            # randomly select one instruction from list
            instruction = self.instruction_templates[
                np.random.choice(len(self.instruction_templates))
            ]

            if self.task in TEXT2MOL_BENCHMARKS:
                """
                "chebi-20-text2mol",
                "smol-name_conversion-i2s",
                "smol-name_conversion-i2f",
                "smol-molecule_generation",
                """
                s_token, e_token = (
                    added_tokens.IUPAC
                    if self.task
                    in ["smol-name_conversion-i2s", "smol-name_conversion-i2f"]
                    else added_tokens.DESCRIPTION
                )

                description = raw_input

                instruction += "\n" + s_token + description + e_token
                graph = smiles2data(
                    "CCCC"
                )  # null smiles, just input for batch processing
                input_mol_string = "<None>"
            elif self.task in MOL2TEXT_BENCHMARKS:
                """
                "chebi-20-mol2text",
                "smol-name_conversion-s2f",
                "smol-name_conversion-s2i",
                "smol-molecule_captioning",
                """

                input_mol_string = raw_input
                smiles = sf.decoder(input_mol_string)
                graph = smiles2data(smiles)
            elif self.task in REACTION_BENCHMARKS:
                input_mol_string = raw_input
                smiles = sf.decoder(input_mol_string)
                graph = smiles2data(smiles)

            label = wrap_label(label, self.task)
            input_mol_string = (
                added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
            )
            output = {
                "graph": graph,
                "label": label,
                "input_mol_string": input_mol_string,
                "instruction": instruction,
            }

            return output
        except Exception as e:
            return e

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
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index":
            return self.x.size(0)
        if key == "additional_edge_index":
            return self.additional_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


# Initialize with the data_list from ConcatDataset
class Mol_LLM_Dataset(InMemoryDataset):
    def __init__(
        self,
        root,
        filename,
        transform=None,
        pre_transform=None,
        resize=None,
        args=None,
    ):
        self.args = args
        self.filename = filename  # raw_file_names and processed_file_names use this
        self.start, self.end = added_tokens.SELFIES
        self.resize = resize
        self.task = "_".join(filename.split("_")[:-1])
        self.split = filename.split("_")[-1]
        self.target_benchmarks = self.get_target_benchmarks()
        super(Mol_LLM_Dataset, self).__init__(root, transform, pre_transform)
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

    # if not all the raw files are exists, download the dataset
    @property
    def raw_file_names(self):
        return [f"{task}_{self.split}.pth" for task in self.target_benchmarks]

    @property
    def processed_file_names(self):
        return f"{self.filename}.pt"

    def get_target_benchmarks(self):
        return self.args.target_benchmarks[self.task]

    def get_dataset(self, task_name):
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)

        # get dataset from deepchem
        if task_name == "bace":
            loading_fn = dc.molnet.load_bace_classification
        elif task_name in [
            "bbbp",
            "clintox",
            "toxcast",
            "sider",
            "tox21",
            "hiv",
            "lipo",
        ]:
            loading_fn = getattr(dc.molnet, f"load_{task_name}")
        elif task_name == "esol":
            loading_fn = dc.molnet.load_delaney
        elif "chebi-20" in task_name:
            dataset = load_dataset("liupf/ChEBI-20-MM")
            train_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            tasks = [task_name]

        # mol-instruction datasets
        elif task_name in [
            "chebi-20-text2mol",
            "chebi-20-mol2text",
            "reagent_prediction",
            "forward_reaction_prediction",
            "retrosynthesis",
            "qm9_homo",
            "qm9_lumo",
            "qm9_homo_lumo_gap",
        ]:
            mol_instruction_dataset = load_dataset(
                "zjunlp/Mol-Instructions",
                "Molecule-oriented Instructions",
                trust_remote_code=True,
            )
            if "qm9_" in task_name:
                dataset = mol_instruction_dataset["property_prediction"]
                subtask_name = task_name.split("_")[1]
                subtask_instruction_templates = getattr(instructions, subtask_name)
                dataset = dataset.filter(
                    lambda x: x["instruction"] in subtask_instruction_templates
                )
            else:
                dataset = mol_instruction_dataset[task_name]

            train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
            split = train_dataset.train_test_split(test_size=0.02, shuffle=True)
            train_dataset, valid_dataset = split["train"], split["test"]

            test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
            tasks = [task_name]
        elif "smol" in task_name:
            # smol_dataset = load_dataset("osunlp/SMolInstruct", use_selfies=True)
            smol_dataset = load_dataset(
                "osunlp/SMolInstruct",
                use_selfies=True,
                insert_core_tags=False,  # loada data w/o core tags such as <SELFIES>, </SELFIES>
            )
            _task = re.sub("smol-", "", task_name)  # remove smol- from smol-<task_name>

            # DEBUG: to avoid lengthy processing time
            train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
            valid_dataset = smol_dataset["validation"].filter(
                lambda x: x["task"] == _task
            )
            test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
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

    def get_dataset(self, task_name):
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)

        # get dataset from deepchem
        if task_name == "bace":
            loading_fn = dc.molnet.load_bace_classification
        elif task_name in [
            "bbbp",
            "clintox",
            "toxcast",
            "sider",
            "tox21",
            "hiv",
            "lipo",
        ]:
            loading_fn = getattr(dc.molnet, f"load_{task_name}")
        elif task_name == "esol":
            loading_fn = dc.molnet.load_delaney
        elif "chebi-20" in task_name:
            dataset = load_dataset("liupf/ChEBI-20-MM")
            train_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            tasks = [task_name]

        # mol-instruction datasets
        elif task_name in [
            "chebi-20-text2mol",
            "chebi-20-mol2text",
            "reagent_prediction",
            "forward_reaction_prediction",
            "retrosynthesis",
            "qm9_homo",
            "qm9_lumo",
            "qm9_homo_lumo_gap",
        ]:
            mol_instruction_dataset = load_dataset(
                "zjunlp/Mol-Instructions",
                "Molecule-oriented Instructions",
                trust_remote_code=True,
            )
            if "qm9" in task_name:
                dataset = mol_instruction_dataset["property_prediction"]
                subtask_name = task_name.split("_")[1]
                subtask_instruction_templates = getattr(instructions, subtask_name)
                dataset = dataset.filter(
                    lambda x: x["instruction"] in subtask_instruction_templates
                )
            else:
                dataset = mol_instruction_dataset[task_name]

            train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
            split = train_dataset.train_test_split(test_size=0.02, shuffle=True)
            train_dataset, valid_dataset = split["train"], split["test"]

            test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
            tasks = [task_name]
        elif "smol" in task_name:
            # smol_dataset = load_dataset("osunlp/SMolInstruct", use_selfies=True)
            smol_dataset = load_dataset(
                "osunlp/SMolInstruct",
                use_selfies=True,
                insert_core_tags=False,  # loada data w/o core tags such as <SELFIES>, </SELFIES>
            )
            _task = re.sub("smol-", "", task_name)  # remove smol- from smol-<task_name>

            # DEBUG: to avoid lengthy processing time
            train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
            valid_dataset = smol_dataset["validation"].filter(
                lambda x: x["task"] == _task
            )
            test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
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

    def download(self):
        # subtask index is necessary when loading clintox from deepchem
        # TODO: deprecate this lengthy hardcoded list
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
            "chebi-20-mol2text": [0],
            "chebi-20-text2mol": [0],
            "smol-molecule_captioning": [0],
            "smol-molecule_generation": [0],
            "smol-name_conversion-s2f": [0],
            "smol-name_conversion-s2i": [0],
            "smol-name_conversion-i2s": [0],
            "smol-name_conversion-i2f": [0],
            "smol-forward_synthesis": [0],
            "smol-retrosynthesis": [0],
        }

        # leave task only if it is in target_benchmarks
        task_subtask_pairs = [
            (task, subtask) if task in self.target_benchmarks else None
            for task, subtasks in task_subtask_lists.items()
            for subtask in subtasks
        ]
        # remove None
        task_subtask_pairs = [t for t in task_subtask_pairs if t]

        self.task_subtask_pairs = []
        for t in task_subtask_pairs:
            task_name = t[0]
            if (
                os.path.exists(f"{self.raw_dir}/{task_name}_val.pth")
                and os.path.exists(f"{self.raw_dir}/{task_name}_test.pth")
                and os.path.exists(f"{self.raw_dir}/{task_name}_train.pth")
            ):
                print(f"{task_name} already exists")
            else:
                self.task_subtask_pairs.append(t)

        multi_task_datasets = {}
        for task_subtask_pair in self.task_subtask_pairs:
            task_name = task_subtask_pair[0]
            new_dataset = self.get_dataset(task_name=task_name)
            multi_task_datasets[task_name] = new_dataset

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
            if task_name in [
                "bace",
                "bbbp",
                "clintox",
                "toxcast",
                "sider",
                "tox21",
                "hiv",
                "lipo",
                "esol",
            ]:
                valid_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    subtask_idx=subtask_idx,
                )
                test_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    subtask_idx=subtask_idx,
                    train_dataset=MoleculeNetDatasetDeepChem(
                        data=data_split[0],
                        task_subtask_pair=task_subtask_pair,
                        subtask_idx=subtask_idx,
                    ),
                )
            elif task_name in ["chebi-20-mol2text", "chebi-20-text2mol"]:
                valid_dataset = ChEBIDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                )
                test_dataset = ChEBIDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                )
                train_dataset = ChEBIDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                )
            # qm9 in regression benchmark is processed via MolInstructionDataset
            elif task_name in [
                "chebi-20-text2mol",
                "chebi-20-mol2text",
                "reagent_prediction",
                "forward_reaction_prediction",
                "retrosynthesis",
                "qm9_homo",
                "qm9_lumo",
                "qm9_homo_lumo_gap",
            ]:
                if not os.path.exists(f"{self.raw_dir}/{task_name}_val.pth"):
                    valid_dataset = MolInstructionDatset(
                        data=data_split[1],
                        task_subtask_pair=task_subtask_pair,
                    )
                if not os.path.exists(f"{self.raw_dir}/{task_name}_test.pth"):
                    test_dataset = MolInstructionDatset(
                        data=data_split[2],
                        task_subtask_pair=task_subtask_pair,
                    )
                if not os.path.exists(f"{self.raw_dir}/{task_name}_train.pth"):
                    train_dataset = MolInstructionDatset(
                        data=data_split[0],
                        task_subtask_pair=task_subtask_pair,
                    )
            elif task_name in [
                "smol-molecule_captioning",
                "smol-molecule_generation",
                "smol-name_conversion-s2f",
                "smol-name_conversion-s2i",
                "smol-name_conversion-i2s",
                "smol-name_conversion-i2f",
                "smol-forward_synthesis",
                "smol-retrosynthesis",
            ]:
                # data leakage inspection between smol-instruct dataset and mol-instruction dataset
                # maintain mol-instruction testset without leakage by excluding smol-instruct trainset

                if not os.path.exists(f"{self.raw_dir}/{task_name}_val.pth"):
                    valid_dataset = SMolInstructDataset(
                        data=data_split[1],
                        task_subtask_pair=task_subtask_pair,
                    )
                if not os.path.exists(f"{self.raw_dir}/{task_name}_test.pth"):
                    test_dataset = SMolInstructDataset(
                        data=data_split[2],
                        task_subtask_pair=task_subtask_pair,
                    )
                if not os.path.exists(f"{self.raw_dir}/{task_name}_train.pth"):
                    train_dataset = SMolInstructDataset(
                        data=data_split[0],
                        task_subtask_pair=task_subtask_pair,
                    )
            if valid_dataset is not None:
                torch.save(valid_dataset, f"{self.raw_dir}/{task_name}_val.pth")
            if test_dataset is not None:
                torch.save(test_dataset, f"{self.raw_dir}/{task_name}_test.pth")
            if train_dataset is not None:
                torch.save(train_dataset, f"{self.raw_dir}/{task_name}_train.pth")
        return

    def process(self):
        # Process data_list and store in `self.data` and `self.slices`
        raw_data_list = []
        for task in self.target_benchmarks:
            if task in [
                "smol-forward_synthesis",
                "smol-retrosynthesis",
            ] and self.split in ["val", "test"]:
                continue
            raw_data_list.extend(
                list(
                    torch.load(
                        f"{self.raw_dir}/{task}_{self.split}.pth", map_location="cpu"
                    )
                )
            )

        # filter out duplicated data in train and test set for smol-forward_synthesis and smol-retrosynthesis
        if self.split == "train" and "smol-forward_synthesis" in self.target_benchmarks:
            test_dataset = torch.load(
                f"{self.raw_dir}/forward_reaction_prediction_test.pth"
            )
            raw_data_list = filter_duplication(raw_data_list, test_dataset)
        elif self.split == "train" and "smol-retrosynthesis" in self.target_benchmarks:
            test_dataset = torch.load(f"{self.raw_dir}/retrosynthesis_test.pth")
            raw_data_list = filter_duplication(raw_data_list, test_dataset)

        data_list = []
        count_fail_conversion = 0
        iter_bar = tqdm(range(len(raw_data_list)))
        for i in iter_bar:
            iter_bar.set_description(
                f"{self.filename}|Num fail: {count_fail_conversion}|Ratio fail: {count_fail_conversion/(i+1)}"
            )
            # graph, label, input_mol_string, task_subtask_pair, instruction
            instance = raw_data_list[i]
            try:
                if "long" in self.filename:
                    # batch processing requires uniform data structure.
                    # for reagent prediction, the input is a pair of graphs
                    # input string: reactant>>product / output string: reagent
                    # maps reactant: first graph, product: second graph
                    if isinstance(instance[0], list):
                        pass
                    # for other tasks, the input is a single graph, but convert the single graph to a pair of graphs
                    # wit dummy graph corresponding to 'CCCC' for batch processing
                    else:
                        dummy_graph = smiles2data("CC")
                        instance = [
                            [instance[0], dummy_graph],
                            instance[1],
                            instance[2],
                            instance[3],
                            instance[4],
                        ]

                    data = PairData(
                        x=instance[0][0].x,
                        edge_index=instance[0][0].edge_index,
                        edge_attr=instance[0][0].edge_attr,
                        additional_x=instance[0][1].x,
                        additional_edge_index=instance[0][1].edge_index,
                        additional_edge_attr=instance[0][1].edge_attr,
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


def filter_duplication(train_dataset, test_dataset):
    import multiprocessing as mp

    num_procs = 200
    dup_idx = mp.Manager().list()
    procs = []
    # mol_strings
    train_data = [instance[2] for instance in train_dataset]
    test_data = test_dataset[:][2]
    indices = np.arange(len(train_data))
    chuncked_idx = np.array_split(indices, num_procs)
    for i in range(num_procs):
        proc = mp.Process(
            target=check_duplication,
            args=(train_data, test_data, chuncked_idx[i], dup_idx),
        )
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    dup_idx = list(dup_idx)
    print(f"Number of duplicated data: {len(dup_idx)}")
    # remove data instance corresponding to duplicated index from train_dataset
    train_idxs = np.arange(len(train_dataset))
    train_idxs = np.delete(train_idxs, dup_idx)
    filtered_train_dataset = Subset(train_dataset, train_idxs)
    return filtered_train_dataset


def check_duplication(train_data, test_data, train_idxs, dup_idx):
    iter_bar = tqdm(range(len(train_idxs)))
    checked_dup = []
    for i in tqdm(iter_bar):
        if train_data[train_idxs[i]] in test_data:
            checked_dup.append(train_idxs[i])
    dup_idx.extend(checked_dup)


if __name__ == "__main__":
    dm = Stage3DM(
        mode="pretrain",
        num_workers=0,
        batch_size=256,
        root="data/",
        text_max_len=128,
        tokenizer=None,
    )
