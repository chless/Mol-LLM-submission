# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import random
import re

import deepchem as dc
import numpy as np
import selfies as sf
import torch
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
from typing import List, Dict, Any

from data_provider import instructions
import model.added_tokens as added_tokens
from pytorch_lightning import LightningDataModule
from transformers.tokenization_utils_base import BatchEncoding
import ast


# we split individual characters inside special tokens like [START_DNA]
# TODO: change this ugly I_SMILES things to regular special token, and add the special token to vocab whichever LLM
# CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")
CUSTOM_SEQ_RE = re.compile(r"(<SELFIES>)(.*?)(</SELFIES>)")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.


def prepare_tokenized_instance(
    data,
    label,
    input_mol_string,
    task_subtask_pair,
    instruction,
    tokenizer,
    fit_llm_input_convention,
    fit_llm_output_convention,
):
    llm_prompt = prepare_llm_input(
        mol_string=input_mol_string,
        instruction=instruction,
        mol_ph=tokenizer.mol_ph_token,
        mol_representation="string_only",
        fit_llm_input_convention=fit_llm_input_convention,
    )

    label = fit_llm_output_convention(label)

    # NOTE: getting tensor is faster that getting list, but become problematic when using collate, due to different tensor size
    input_tokens = tokenizer(
        text=llm_prompt + label,
        return_attention_mask=False,
        return_length=True,
        return_token_type_ids=False,
    )
    prompt_tokens = tokenizer(llm_prompt, return_length=True)

    if isinstance(prompt_tokens.length, list):
        prompt_tokens_length = prompt_tokens.length[0]
    elif isinstance(prompt_tokens.length, int):
        prompt_tokens_length = prompt_tokens.length
    else:
        raise NotImplementedError

    target_text = prompt_tokens_length * tokenizer.pad_token + label

    # TODO: later, when start training graph modality with sequence packing, change the PairData to PackedData
    prepared_instance = PairData(
        **data,
        input_text=llm_prompt + label,
        target_text=target_text,
        prompt_text=llm_prompt,
        task_subtask_pair=task_subtask_pair,
        input_ids=input_tokens.input_ids,
    )
    return prepared_instance


def prepare_llm_input(
    mol_string,
    instruction,
    mol_ph,
    mol_representation,
    fit_llm_input_convention=None,
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
        # name conversion tasks are included
        # if you use LlaSMol instruction for M2T, replace <INPUT> with mol_string
        llm_prompt = instruction.replace("<INPUT>", mol_string_converted)
    elif not "<None>" in mol_string:
        llm_prompt = instruction + mol_string_converted
    else:
        llm_prompt = instruction

    llm_prompt = fit_llm_input_convention(llm_prompt)
    return llm_prompt


def pack_data_points(data_list):
    # TODO: implement proper graph handling when graph training
    packed_x = data_list[0].x
    packed_edge_index = data_list[0].edge_index
    packed_edge_attr = data_list[0].edge_attr
    packed_additional_x = data_list[0].additional_x
    packed_additional_edge_index = data_list[0].additional_edge_index
    packed_additional_edge_attr = data_list[0].additional_edge_attr

    packed_input_texts = [instance.input_text for instance in data_list]
    packed_input_texts = "".join(packed_input_texts)

    packed_target_texts = [instance.target_text for instance in data_list]
    packed_target_texts = "".join(packed_target_texts)
    packed_task_subtask_pair = [instance.task_subtask_pair for instance in data_list]

    packed_instance = PairData(
        x=packed_x,
        edge_index=packed_edge_index,
        edge_attr=packed_edge_attr,
        additional_x=packed_additional_x,
        additional_edge_index=packed_additional_edge_index,
        additional_edge_attr=packed_additional_edge_attr,
        input_text=packed_input_texts,
        target_text=packed_target_texts,
        task_subtask_pair=packed_task_subtask_pair,
    )
    return packed_instance


def group_data_idx_by_length(
    lengths: List[int], max_length: int, max_size: int = -1, max_instance_length=512
) -> List[List[int]]:
    """given lengths of data points, we merge consecutive data points into a new data point, as long as the concatenated length is less than max_length
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        max_size: if != -1; the maximum number of consecutive items being merged; max_size: -1 --> no limit for number of items being merged

    max_size: the maximum number of data points being merged
    For example, lengths=[1, 3, 2, 2, 6, 4, 2, 6, 5]; max_length=10
    if max_size=-1 --> [[0,1,2,3], [4, 5], [6,7], [8]]
    if max_size=3 --> [[0,1,2], [3,4], [5, 6], [7], [8]]

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    result = []
    current_concatenated_length = 0
    current_list = []
    count_length_cutoff = 0
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length > max_instance_length:
            count_length_cutoff += 1
            continue

        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == (
        len(lengths) - count_length_cutoff
    )
    return result


import multiprocessing as mp


def filter_duplication(train_dataset, test_dataset, num_procs=20):

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


class DataCollater:
    def __init__(
        self,
        tokenizer,
        max_length,
        truncation,
        padding,
        mode,
        apply_sequence_packing=False,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.truncation = bool(truncation)
        self.padding = padding
        self.mode = mode
        self.apply_sequence_packing = apply_sequence_packing

    def __call__(self, batch):
        target_texts = [instance.target_text for instance in batch]
        if self.mode == "eval":
            input_texts = [instance.prompt_text for instance in batch]
        else:
            input_texts = [instance.input_text for instance in batch]

        if isinstance(batch[0], PairData):
            additional_batch = torch.tensor([], dtype=torch.int64)
            for i in range(len(batch)):
                additional_num_nodes = batch[i].additional_x.size(0)
                additional_node_indexing_tensor = torch.tensor(
                    [i] * additional_num_nodes, dtype=torch.int64
                )
                additional_batch = torch.cat(
                    (additional_batch, additional_node_indexing_tensor), 0
                )

        batch = self.collater(batch)

        if isinstance(batch, PairData):
            batch.additional_batch = additional_batch

        self.tokenizer.padding_side = "right"
        input_tokens = self.tokenizer(
            text=input_texts,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=False,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_tokens["is_mol_token"] = (
            input_tokens.input_ids == self.tokenizer.mol_token_id
        )

        if self.apply_sequence_packing:
            input_attention_mask = get_attention_mask_for_packed_sequence(
                x=input_tokens.input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            input_tokens["attention_mask"] = input_attention_mask

        target_tokens = self.tokenizer(
            text=target_texts,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=False,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return batch, input_tokens, target_tokens


def get_attention_mask_for_packed_sequence(x, eos_token_id, include_eos: bool = True):
    B, T = x.shape
    eos_idx = (x.view(-1) == eos_token_id).nonzero(as_tuple=True)[0] + include_eos
    eos_idx_expanded = (
        torch.cat([eos_idx, torch.arange(0, B * T + 1, T)]).unique().sort()[0]
    )
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = (
        normalized_idx[1:] - normalized_idx[:-1]
    )  # num of tokens in sequences including eos count
    reps = torch.where(
        reps < 1, normalized_idx[1:], reps
    )  # reps < 1 means the token is the first token of the sequence
    idxs_seq_marked = (
        torch.repeat_interleave(
            normalized_idx[1:], reps
        )  # normalized_idx[1:] represent all the distinct sequences and padding sequences
        .view(B, 1, T)
        .expand(-1, T, -1)
    )
    mask_indices = torch.arange(T).view(1, -1, 1).expand(B, -1, T)
    mask = torch.ones(T, T, dtype=torch.long).tril().expand(B, -1, -1)
    mask = mask.masked_fill(mask_indices >= idxs_seq_marked, 0).unsqueeze(
        1
    )  # to fit the shape [B, 1, T, T]
    return mask


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
    "qm9_additional_label",
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
        tokenizer,
        fit_llm_input_convention,
        fit_llm_output_convention,
        mode: str = "pretrain",
        num_workers: int = 0,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.num_workers = num_workers

        self.fit_llm_input_convention = fit_llm_input_convention
        self.fit_llm_output_convention = fit_llm_output_convention

        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.inference_batch_size = args.inference_batch_size
        self.inference_max_length = args.inference_max_length

        self.mol_representation = args.mol_representation
        self.dataset_split = {"train": None, "val": None, "test": None}

        for split in ["train", "test", "val"]:
            if split == "val":
                resize = args.valset_resize if args.valset_resize > 0 else None
            elif split == "test":
                resize = args.testset_resize if args.testset_resize > 0 else None
            elif split == "train":
                resize = args.trainset_resize if args.trainset_resize > 0 else None
            else:
                raise NotImplementedError

            if split == "val":
                data_split = f"test"
            elif split == "train" and hasattr(self.args, "debug"):
                data_split = f"test"
            elif split == "test" and self.args.test_on_trainset:
                data_split = f"train"
            else:
                data_split = f"{split}"

            self.dataset_split[split] = Mol_LLM_Dataset(
                root=self.args.raw_data_root,
                split=data_split,
                mode=split,
                tokenizer=tokenizer,
                fit_llm_input_convention=self.fit_llm_input_convention,
                fit_llm_output_convention=self.fit_llm_output_convention,
                resize=resize,
                args=self.args,
            )

        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_split["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=DataCollater(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                truncation=self.args.truncation,
                padding=self.args.padding,
                mode="train",
                apply_sequence_packing=self.args.apply_sequence_packing,
            ),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_split["val"],
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=DataCollater(
                tokenizer=self.tokenizer,
                max_length=self.inference_max_length,
                truncation=self.args.truncation,
                padding=self.args.padding,
                mode="eval",
                apply_sequence_packing=False,
            ),
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_split["test"],
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=DataCollater(
                tokenizer=self.tokenizer,
                max_length=self.inference_max_length,
                truncation=self.args.truncation,
                padding=self.args.padding,
                mode="eval",
                apply_sequence_packing=False,
            ),
        )
        return loader


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
            if self.task in ["qm9_additional_label"]:
                subtask_full_name_dict = {
                    "mu": "dipole_moment",
                    "alpha": "isotropic_polarizability",
                    "r2": "electronic_spatial_extent",
                    "zpve": "zero_point_vibrational_energy",
                    "cv": "heat_capacity_298K",
                    "u298": "internal_energy_298K",
                    "h298": "enthalpy_298K",
                    "g298": "free_energy_298K",
                }
                task = self.task.replace("_additional_label", "")
                subtask_full_name = subtask_full_name_dict[self.subtask]
                self.instruction_templates = getattr(
                    instructions, f"{task}_{subtask_full_name}"
                )
            else:
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
        label = self.raw_outputs[index]
        label = wrap_label(label, self.task)
        graph = smiles2data(smiles)
        # randomly select one instruction from list
        instruction = self.instruction_templates[
            np.random.choice(len(self.instruction_templates))
        ]
        return graph, label, input_mol_string, instruction

    def set_necessary_data(self):
        self.raw_inputs = self.data.X
        self.raw_outputs = self.data.y[:, self.subtask_idx]

        self.smiles_list = []
        for mol in self.raw_inputs:
            self.smiles_list.append(Chem.MolToSmiles(mol))

        self.label_list = []
        self.input_mol_string_list = []
        self.graph_list = []
        self.instruction_list = []

        self.count_invalid_smiles = 0

        iter_bar = tqdm(
            range(len(self.raw_inputs)),
            total=len(self.raw_inputs),
            desc=f"{self.task}-{self.subtask_idx}",
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
                f"{self.task}: Invalid smiles ratio: {self.count_invalid_smiles/len(self.raw_inputs)}"
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
    def __init__(self, data, task_subtask_pair, **kwargs):
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
    def __init__(self, data, task_subtask_pair, **kwargs):
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
    def __init__(self, data, task_subtask_pair, **kwargs):
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
        elif key == "additional_edge_index":
            return self.additional_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class PackedData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "edge_index" in key:
            prefix = key.split("edge_index")[0]
            return getattr(self, f"{prefix}x.size")(0)
        return super().__inc__(key, value, *args, **kwargs)

# Initialize with the data_list from ConcatDataset
class Mol_LLM_Dataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split,
        mode,
        tokenizer,
        fit_llm_input_convention,
        fit_llm_output_convention,
        resize=None,
        transform=None,
        pre_transform=None,
        args=None,
    ):
        self.args = args
        self.split = split
        self.mode = mode
        self.tokenizer = tokenizer
        self.start, self.end = added_tokens.SELFIES
        self.resize = resize
        self.task_subtask_dict, self.task_subtask_pairs = self.get_task_subtask_info()
        self.fit_llm_input_convention = fit_llm_input_convention
        self.fit_llm_output_convention = fit_llm_output_convention
        self.apply_sequence_packing = (
            True if self.args.apply_sequence_packing and self.mode == "train" else False
        )
        self.llm_model_name = self.args.llm_model.replace("/", "_")
        self.data_tag = self.args.data_tag

        super(Mol_LLM_Dataset, self).__init__(root, transform, pre_transform)
        print(f"loading dataset {self.processed_file_names}")
        self.load(self.processed_paths[0])
        print(
            f"loaded dataset {self.processed_file_names}| data length: {len(self._data.input_ids)}"
        )

        if mode == 'test':
            self.task_subtask_name_pairs = list(set(self.data.task_subtask_pair))

        self.set_data_indices()
        self.shuffle_data_indices()

        # __len__ is not prepared for sequence packing case, so use len(self.indices) instead
        if self.resize and len(self.indices) > self.resize:
            self.reduce_dataset_size(self.resize)

        if self.apply_sequence_packing:
            self.data_length_list = self.get_data_length_list()
            self.groups = group_data_idx_by_length(
                lengths=self.data_length_list,
                max_length=self.args.max_length,
                max_size=self.args.max_packing_size,
            )

    def __len__(self):
        if self.apply_sequence_packing:
            return len(self.groups)
        else:
            return len(self.indices)

    # access data instance only via self.indices, to achieve fast data shuffling effect by only shuffle access idx
    def get(self, idx):
        accessed_idx = self.indices[idx]
        return super(Mol_LLM_Dataset, self).get(accessed_idx)

    def __getitem__(self, index):
        if self.apply_sequence_packing:
            group = self.groups[index]
            data_list = [self.get(i) for i in group]
            data = pack_data_points(data_list)
        else:
            data = self.get(index)
        return data

    # access data instance only via self.indices, which ensure fast shuffling for every training epoch
    def set_data_indices(self):
        self.data_len = len(self._data.input_ids)
        self.indices = list(range(self.data_len))

    # shuffle only the indices for fast shuffling
    def shuffle_data_indices(self):
        self.indices = torch.randperm(self.data_len).tolist()

    def reduce_dataset_size(self, new_size):
        data_list = [self.get(i) for i in range(new_size)]
        self.data, self.slices = self.collate(data_list)
        self.set_data_indices()

    def get_data_length_list(self):
        data_length_list = []
        for i in range(self.data_len):
            data_length = len(self.get(i).input_ids)
            data_length_list.append(data_length)
        return data_length_list

    def get_task_subtask_info(self):
        task_subtask_dict = {}
        for task in self.args.target_benchmarks:
            if isinstance(task, str):
                task_subtask_dict[task] = [0]
            else:
                task_subtask_dict.update(task)

        task_subtask_pairs = [
            (task, subtask)
            for task, subtasks in task_subtask_dict.items()
            for subtask in subtasks
        ]
        return task_subtask_dict, task_subtask_pairs

    # if not all the raw files are exists, download the dataset
    @property
    def raw_file_names(self):
        raw_files = [
            f"{task}_subtask-{subtask_idx}_{self.split}.pth"
            for task, subtask_idx in self.task_subtask_pairs
        ]
        return raw_files

    @property
    def processed_file_names(self):
        return f"{self.llm_model_name}_{self.data_tag}_{self.split}.pt"

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
        elif task_name == "qm9_additional_label":
            loading_fn = dc.molnet.load_qm9
        elif "chebi-20" in task_name:
            dataset = load_dataset("liupf/ChEBI-20-MM", trust_remote_code=True)
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
            smol_dataset = load_dataset(
                "osunlp/SMolInstruct",
                use_selfies=True,
                insert_core_tags=False,  # loada data w/o core tags such as <SELFIES>, </SELFIES>
                trust_remote_code=True,
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
            and task_name not in ["qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap"]
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
        downloading_task_subtask_pairs = []
        for task_subtask_pair in self.task_subtask_pairs:
            task, subtask_idx = task_subtask_pair
            if (
                os.path.exists(f"{self.raw_dir}/{task}_subtask-{subtask_idx}_val.pth")
                and os.path.exists(
                    f"{self.raw_dir}/{task}_subtask-{subtask_idx}_test.pth"
                )
                and os.path.exists(
                    f"{self.raw_dir}/{task}_subtask-{subtask_idx}_train.pth"
                )
            ):
                print(f"{task}_{subtask_idx} already exists")
            else:
                downloading_task_subtask_pairs.append(task_subtask_pair)

        multi_task_datasets = {}
        for task_subtask_pair in downloading_task_subtask_pairs:
            task_name = task_subtask_pair[0]
            if task_name not in multi_task_datasets:
                new_dataset = self.get_dataset(task_name=task_name)
                multi_task_datasets[task_name] = new_dataset
            else:
                pass

        for task_subtask_pair in tqdm(
            downloading_task_subtask_pairs, desc="Downloading task_subtask_pairs"
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
                "qm9_additional_label",
            ]:
                dataset = MoleculeNetDatasetDeepChem
            elif task_name in ["chebi-20-mol2text", "chebi-20-text2mol"]:
                dataset = ChEBIDatset
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
                dataset = MolInstructionDatset
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
                dataset = SMolInstructDataset

            valid_dataset = dataset(
                data=data_split[1],
                task_subtask_pair=task_subtask_pair,
                subtask_idx=subtask_idx,
            )
            test_dataset = dataset(
                data=data_split[2],
                task_subtask_pair=task_subtask_pair,
                subtask_idx=subtask_idx,
            )
            train_dataset = dataset(
                data=data_split[0],
                task_subtask_pair=task_subtask_pair,
                subtask_idx=subtask_idx,
            )

            if (
                hasattr(self.args, "duplication_check_train")
                and task_name in self.args.duplication_check_train
            ):
                train_task_idx = self.args.duplication_check_train.index(task_name)
                test_task = self.args.duplication_check_test[train_task_idx]

                print(
                    f"Checking duplication between train: {task_name} and test: {test_task}"
                )
                test_data = torch.load(f"{self.raw_dir}/{test_task}_subtask-0_test.pth")
                # get data_list from train_data
                train_dataset = filter_duplication(
                    train_dataset, test_data, num_procs=30
                )
            torch.save(
                valid_dataset,
                f"{self.raw_dir}/{task_name}_subtask-{subtask_idx}_val.pth",
            )
            torch.save(
                test_dataset,
                f"{self.raw_dir}/{task_name}_subtask-{subtask_idx}_test.pth",
            )
            torch.save(
                train_dataset,
                f"{self.raw_dir}/{task_name}_subtask-{subtask_idx}_train.pth",
            )

    def process(self):
        # load raw datasets in target_benchmarks
        raw_data_list = []
        iter_bar = tqdm(
            range(len(self.task_subtask_pairs)),
            desc="Loading raw data",
            total=len(self.task_subtask_pairs),
        )
        for i in iter_bar:
            task, subtask_idx = self.task_subtask_pairs[i]
            # data leakage check: not use val, test set of the tasks subject to duplication check
            if (
                hasattr(self.args, "duplication_check_train")
                and task in self.args.duplication_check_train
                and self.split in ["val", "test"]
            ):
                continue
            # data leakage check: not use duplicated train idx of the tasks subject to duplication check
            # TODO: move this phase to download method
            elif (
                hasattr(self.args, "duplication_check_train")
                and task in self.args.duplication_check_train
                and self.split == "train"
            ):
                # find train task idx in duplication_check_train
                train_task_idx = self.args.duplication_check_train.index(task)
                test_task = self.args.duplication_check_test[train_task_idx]
                print(
                    f"Checking duplication between train: {task} and test: {test_task}"
                )

                # not permit same molecule across train and test set
                test_data = torch.load(f"{self.raw_dir}/{test_task}_subtask-0_test.pth")
                train_data = list(
                    torch.load(f"{self.raw_dir}/{task}_subtask-{subtask_idx}_train.pth")
                )
                before_len = len(train_data)
                raw_data = filter_duplication(train_data, test_data, num_procs=50)
                print(
                    f"Number of removed data for duplication: {before_len - len(raw_data)}"
                )
            else:
                iter_bar.set_description(
                    f"Loading {task}_subtask-{subtask_idx}_{self.split}"
                )
                raw_data = list(
                    torch.load(
                        f"{self.raw_dir}/{task}_subtask-{subtask_idx}_{self.split}.pth"
                    )
                )
            raw_data_list.extend(raw_data)

        # process raw_data_list
        processed_data_list = []
        count_failed_conversion = 0

        iter_bar = tqdm(range(len(raw_data_list)))
        for i in iter_bar:
            iter_bar.set_description(
                f"{self.llm_model_name}-{self.split}|Num fail: {count_failed_conversion}|Ratio fail: {count_failed_conversion/(i+1)}"
            )
            # graph, label, input_mol_string, task_subtask_pair, instruction
            instance = raw_data_list[i]
            try:
                # batch processing requires uniform data structure.
                # for reagent prediction, the input is a pair of graphs
                # input string: reactant>>product / output string: reagent
                # maps reactant: first graph, product: second graph
                if isinstance(instance[0], list):
                    pass
                # for other tasks, the input is a single graph, but convert the single graph to a pair of graphs
                # wit dummy graph corresponding to 'CC' for uniform data structure with reagent prediction
                else:
                    dummy_graph = smiles2data("CC")
                    instance = [
                        [instance[0], dummy_graph],
                        instance[1],
                        instance[2],
                        instance[3],
                        instance[4],
                    ]

                data = prepare_tokenized_instance(
                    data=PairData(
                        x=instance[0][0].x,
                        edge_index=instance[0][0].edge_index,
                        edge_attr=instance[0][0].edge_attr,
                        additional_x=instance[0][1].x,
                        additional_edge_index=instance[0][1].edge_index,
                        additional_edge_attr=instance[0][1].edge_attr,
                    ),
                    label=instance[1],
                    input_mol_string=instance[2],
                    task_subtask_pair=instance[3],
                    instruction=instance[4],
                    tokenizer=self.tokenizer,
                    fit_llm_input_convention=self.fit_llm_input_convention,
                    fit_llm_output_convention=self.fit_llm_output_convention,
                )

                processed_data_list.append(data)
            except:
                count_failed_conversion += 1
                continue

        self.save(
            processed_data_list,
            os.path.join(
                self.processed_dir,
                f"{self.llm_model_name}_{self.data_tag}_{self.split}.pt",
            ),
        )


if __name__ == "__main__":
    dm = Stage3DM(
        mode="pretrain",
        num_workers=0,
        batch_size=256,
        root="data/",
        text_max_len=128,
        tokenizer=None,
    )
