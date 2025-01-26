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
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
from typing import List, Dict, Any

import instructions_smol
import model.added_tokens as added_tokens
from pytorch_lightning import LightningDataModule
import pandas as pd
import datasets

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.


def prepare_text_data(
    data,
    label,
    input_mol_string,
    task_subtask_pair,
    instruction,
    fit_llm_input_convention,
    fit_llm_output_convention,
):

    if (
        "<INPUT>" in instruction and not "<None>" in input_mol_string
    ):  # for LlaSMol whose input contains <INPUT>
        # name conversion tasks are included
        # if you use LlaSMol instruction for M2T, replace <INPUT> with mol_string
        llm_prompt = instruction.replace("<INPUT>", input_mol_string)
    elif not "<None>" in input_mol_string:
        llm_prompt = instruction + input_mol_string
    else:
        llm_prompt = instruction

    llm_prompt = fit_llm_input_convention(llm_prompt)

    label = fit_llm_output_convention(label)

    prepared_instance = PairData(
        **data,
        prompt_text=llm_prompt,
        target_text=label,
        task_subtask_pair=task_subtask_pair,
    )
    return prepared_instance


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


def filter_duplication(subject_dataset, subject_list, reference_list, num_procs=20):

    dup_idx = mp.Manager().list()
    procs = []

    indices = np.arange(len(subject_list))
    chuncked_idx = np.array_split(indices, num_procs)
    for i in range(num_procs):
        proc = mp.Process(
            target=check_duplication,
            args=(subject_list, reference_list, chuncked_idx[i], dup_idx),
        )
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    dup_idx = list(dup_idx)

    # remove data instance corresponding to duplicated index from train_dataset
    subject_idxs = np.arange(len(subject_dataset))
    subject_idxs = np.delete(subject_idxs, dup_idx)
    filtered_train_dataset = Subset(subject_dataset, subject_idxs)
    return filtered_train_dataset


def check_duplication(subject_list, reference_list, subject_idxs, dup_idx):
    iter_bar = tqdm(range(len(subject_idxs)))
    checked_dup = []
    for i in tqdm(iter_bar):
        if subject_list[subject_idxs[i]] in reference_list:
            checked_dup.append(subject_idxs[i])
    dup_idx.extend(checked_dup)


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
    "tox21",  # 12 tasks # molca
    "toxcast",  # 617 # molca
    "smol-property_prediction-bbbp",  # 1 task # molca, biot5+, instructmol, llasmol
    "smol-property_prediction-clintox",  # 2 tasks # molca, biot5+, llasmol
    "smol-property_prediction-hiv",  # 1 tasks # biot5+, instructmol, llasmol
    "smol-property_prediction-sider",  # 27 # molca, llasmol
]
REGRESSION_BENCHMARKS = [
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
    "qm9_additional_label",
    "smol-property_prediction-esol",  # 1 task # llasmol
    "smol-property_prediction-lipo",  # 1 task # llasmol
    "hopv"
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
        if isinstance(label, str):
            if "true" in label.lower() or "yes" in label.lower():
                label = "True"
            elif "false" in label.lower() or "no" in label.lower():
                label = "False"
            else:
                raise NotImplementedError(
                    f"Label: {label} is not supported in classification task"
                )
            label = label_tokens[0] + label + label_tokens[1]
        elif isinstance(label, list):
            label_language = ", ".join(label)
            label_boolean = "True" * len(label)
            label = label_language + label_tokens[0] + label_boolean + label_tokens[1]
        else:
            label = "True" if label else "False"
            label = label_tokens[0] + label + label_tokens[1]
        return label
    elif task in REGRESSION_BENCHMARKS:
        if isinstance(label, float):
            label = "{:.10f}".format(label)
        else:
            label = format(float(label), ".10f")

        # force to predict the sign of label first
        if "-" not in label and "+" not in label:
            label = "+" + label
        # unify the length of label to 7
        label = label[:7]
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
            self.instruction_templates = getattr(instructions_smol, self.task)
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
                    instructions_smol, f"{task}_{subtask_full_name}"
                )
                assert len(self.instruction_templates) > 1, "Instruction is not enough"
            else:
                self.instruction_templates = getattr(instructions_smol, f"{self.task}_{self.subtask}".lower())
            self.label_tokens = added_tokens.FLOAT
        else:
            raise NotImplementedError

        self.set_necessary_data()

    def get_necessary_data(self, index):
        instruction = np.random.choice(self.instruction_templates)
        smiles = self.smiles_list[index]
        # set molecule string representation as selfies
        input_mol_string = sf.encoder(smiles)
        input_mol_string = (
            added_tokens.SELFIES[0] + input_mol_string + added_tokens.SELFIES[1]
        )
        if self.subtask_idx == "multi_label_classification":
            label = self.raw_outputs[index]
            label = [self.label_full_name[i] for i in range(len(label)) if label[i]]
            if len(label) == 0:
                label = "No toxicity identified. " + wrap_label("False", self.task)
            else:
                label = wrap_label(label, self.task)
        else:
            label = self.raw_outputs[index]
            label = wrap_label(label, self.task)

        graph = smiles2data(smiles)
        # randomly select one instruction from list
        return graph, label, input_mol_string, instruction

    def set_label_fullname(self):
        self.label_full_name = None
        if self.task == "tox21":
            self.label_full_name = [
                "androgen receptor, full (AR, full)",  # AR
                "androgen receptor, LBD (AR, LBD)",  # AR, LBD
                "aryl hydrocarbon receptor (AhR)",  # AhR
                "aromatase",
                "estrogen receptor alpha, full (ER, full)",  # ER
                "estrogen receptor alpha, LBD (ER, LBD)",  # ER, LBD
                "peroxisome proliferator-activated receptor gamma (PPAR-gamma)",  # PPAR-gamma
                "nuclear factor (erythroid-derived 2)-like 2/antioxidant responsive element (Nrf2/ARE)",
                "ATPase family AAA domain containing 5 (ATAD5)",
                "heat shock factor response element (HSE)",  # HSE
                "mitochondrial membrane potential (MMP)",  # MMP
                "tumor suppressor protein p53",
            ]

    def set_necessary_data(self):
        self.raw_inputs = self.data.X
        if self.subtask_idx == "multi_label_classification":
            self.set_label_fullname()
            self.raw_outputs = self.data.y
        else:
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
        if self.task == "bace":
            self.input_list = self.data["SELFIES"][:]
            self.label_list = self.data["label"][:]
        else:
            self.input_list = self.data["input"][:]
            self.label_list = self.data["output"][:]
        self.instruction_templates = getattr(instructions_smol, self.task)

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
        instruction = np.random.choice(self.instruction_templates)
        input = self.input_list[index]  # if mol_string, representation is selfies
        label = self.label_list[index]  # if mol_string, representation is selfies

        if self.task in REACTION_BENCHMARKS:
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
        elif self.task in CLASSIFICATION_BENCHMARKS:
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


class ChEBIDataset(Dataset):
    def __init__(self, data, task_subtask_pair, **kwargs):
        self.data = data
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        self.set_necesary_data()

    def set_necesary_data(self):
        self.description_list = self.data["description"]
        self.selfies_list = self.data["SELFIES"]
        if "mol2text" in self.task:
            self.instruction_templates = getattr(
                instructions_smol, "molecule_captioning"
            )
        elif "text2mol" in self.task:
            self.instruction_templates = getattr(
                instructions_smol, "molecule_generation"
            )
        else:
            raise NotImplementedError

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
        instruction = np.random.choice(self.instruction_templates)
        descriptiopn = self.description_list[index]
        selfies = self.selfies_list[index]
        smiles = sf.decoder(selfies)

        if self.task in TEXT2MOL_BENCHMARKS:
            label = selfies
            description = (
                added_tokens.DESCRIPTION[0] + descriptiopn + added_tokens.DESCRIPTION[1]
            )
            instruction = instruction.replace("<INPUT>", description)
            graph = smiles2data(
                "CC"
            )  # null smiles, just input dummy graph for batch processing
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
        if "forward_synthesis" in self.task:
            self.instruction_templates = getattr(
                instructions_smol, "forward_reaction_prediction"
            )
        else:
            self.instruction_templates = getattr(
                instructions_smol, self.task.replace("smol-", "").replace("-", "_")
            )
        self.set_necesary_data()

    def set_necesary_data(self):
        self.semi_colon_count_input = 0
        self.semi_colon_count_label = 0

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
        self.count_invalid_smiles = 0

        for i in iter_bar:
            try:
                graph, label, input_mol_string, instruction = self.get_necessary_data(
                    i, raw_inputs[i], raw_outputs[i]
                )
                self.graph_list.append(graph)
                self.label_list.append(label)
                self.input_mol_string_list.append(input_mol_string)
                self.instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Invalid smiles ratio: {1.0 - len(self.label_list)/len(self.data)}"
            )

    def __len__(self):
        return len(self.label_list)

    def get_necessary_data(self, index, raw_input, raw_output):
        raw_input = raw_input
        label = raw_output

        if ";" in raw_input:
            self.semi_colon_count_input += 1
        if ";" in raw_output:
            self.semi_colon_count_label += 1

        if self.task in TEXT2MOL_BENCHMARKS:
            s_token, e_token = (
                added_tokens.IUPAC
                if self.task in ["smol-name_conversion-i2s", "smol-name_conversion-i2f"]
                else added_tokens.DESCRIPTION
            )
            description = raw_input
            description = s_token + description + e_token
            instruction = np.random.choice(self.instruction_templates)
            instruction = instruction.replace("<INPUT>", description)
            graph = smiles2data(
                "CC"
            )  # null smiles, just input dummy graph for batch processing
            input_mol_string = "<None>"
            label = re.sub(r"\s*;\s*", ".", label)
        elif self.task in REACTION_BENCHMARKS:
            instruction = np.random.choice(self.instruction_templates)
            input_mol_string = raw_input
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        # multi labeled property prediction datasets
        elif self.task in ["smol-property_prediction-sider"]:
            instance_input = self.data[index]["input"]
            assert re.search(r"\[.*\]", instance_input) is not None
            instruction = re.sub(r"\[.*\]", "<INPUT>", instance_input)

            # use re sub to replace ";" with "."
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        elif (
            self.task
            in MOL2TEXT_BENCHMARKS + CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS
        ):
            instruction = np.random.choice(self.instruction_templates)
            # use re sub to replace ";" with "."
            input_mol_string = re.sub(r"\s*;\s*", ".", raw_input)
            smiles = sf.decoder(input_mol_string)
            graph = smiles2data(smiles)
        else:
            raise NotImplementedError(f"Task: {self.task} is not supported")

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

from rdkit import Chem


def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None


import yaml


def get_task_subtask_info(target_benchmarks):
    task_subtask_dict = {}
    for task in target_benchmarks:
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


def get_dataset(task_name, raw_data_root):
    # get dataset from deepchem
    if "smol" in task_name:
        smol_dataset = load_dataset(
            "osunlp/SMolInstruct",
            use_selfies=True,
            insert_core_tags=False,  # loada data w/o core tags such as <SELFIES>, </SELFIES>
            trust_remote_code=True,
        )
        _task = re.sub("smol-", "", task_name)  # remove smol- from smol-<task_name>

        # DEBUG: to avoid lengthy processing time
        train_dataset = smol_dataset["train"].filter(lambda x: x["task"] == _task)
        valid_dataset = smol_dataset["validation"].filter(lambda x: x["task"] == _task)
        test_dataset = smol_dataset["test"].filter(lambda x: x["task"] == _task)
        tasks = [task_name]
    elif task_name in [
        "toxcast",
        "tox21",
        "hopv"
    ]:
        loading_fn = getattr(dc.molnet, f"load_{task_name}")
    elif task_name in ["qm9_additional_label"]:
        loading_fn = dc.molnet.load_qm9
    # TODO: address biot5 datasets
    elif task_name == "bace":
        train_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_bace_train.csv")
        )
        valid_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_bace_valid.csv")
        )
        test_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_bace_test.csv")
        )
        tasks = [task_name]
    elif "chebi-20" in task_name:
        # load data from csv
        train_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_chebi20_train.csv")
        )
        valid_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_chebi20_valid.csv")
        )
        test_dataset = pd.read_csv(
            os.path.join(raw_data_root, "raw/BioT5_chebi20_test.csv")
        )

        tasks = [task_name]

    # mol-instruction datasets
    elif task_name in [
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
            subtask_instruction_templates = getattr(
                instructions_smol, "filtering_template_" + subtask_name
            )
            dataset = dataset.filter(
                lambda x: x["instruction"] in subtask_instruction_templates
            )
            assert len(dataset) > 0, f"len(dataset) = {len(dataset)}"
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
        and task_name not in ["qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap", "bace"]
        and "smol" not in task_name
    ):
        base_path = f"dataset/{task_name}"
        os.makedirs(base_path, exist_ok=True)
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


def from_dict(dict):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Struct(**dict)

if __name__ == "__main__":
    import argparse
    import os
    import random

    # get arg replace_ratio, dataset_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="Mol-LLM/configs/download/")
    parser.add_argument("--config", type=str, default="hopv_homo")
    parser.add_argument("--train_procs", type=int, default=1)
    parser.add_argument("--test_procs", type=int, default=1)
    args = parser.parse_args()

    
    arg_path = os.path.join(args.config_dir, args.config) + ".yaml"
    # read config file
    with open(arg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # convert args to be accessible its values by attributes
    cfg = from_dict(cfg)

    raw_data_root = cfg.raw_data_root

    if not os.path.exists(raw_data_root):
        os.makedirs(raw_data_root)

    start, end = added_tokens.SELFIES
    task_subtask_dict, task_subtask_pairs = get_task_subtask_info(
        cfg.target_benchmarks
    )
    data_tag = cfg.data_tag

    downloading_task_subtask_pairs = []
    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        if os.path.exists(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_train"
        ) and os.path.exists(f"{raw_data_root}/{task}_subtask-{subtask_idx}_test"):
            print(f"{task}_{subtask_idx} already exists")
        else:
            downloading_task_subtask_pairs.append(task_subtask_pair)

    for task_subtask_pair in tqdm(
        downloading_task_subtask_pairs, desc="Downloading task_subtask_pairs"
    ):
        task_name = task_subtask_pair[0]
        new_dataset = get_dataset(task_name=task_name, raw_data_root=raw_data_root)

        subtasks = new_dataset[0]
        subtask_idx = task_subtask_pair[1]
        if subtask_idx == "multi_label_classification":
            task_subtask_pair = f"{task_name}/{subtask_idx}"
        else:
            task_subtask_pair = f"{task_name}/{subtasks[subtask_idx]}"

        data_split = new_dataset[1:]  # train_set, val_set, test_set
        if "smol" in task_name:
            dataset = SMolInstructDataset
        elif task_name in [
            "toxcast",
            "tox21",
            "qm9_additional_label",
            "hopv"
        ]:
            dataset = MoleculeNetDatasetDeepChem
        elif task_name in ["chebi-20-mol2text", "chebi-20-text2mol"]:
            dataset = ChEBIDataset
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
            "bace",
        ]:
            dataset = MolInstructionDatset

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
        dataset_splits = {
            "val": valid_dataset,
            "test": test_dataset,
            "train": train_dataset,
        }

        for split in dataset_splits.keys():
            dataset = dataset_splits[split]
            list_dict_data = []
            for i in range(len(dataset)):
                data = dataset[i]
                dict_data = {
                    "x": data[0].x,
                    "edge_index": data[0].edge_index,
                    "edge_attr": data[0].edge_attr,
                    "label": data[1],
                    "input_mol_string": data[2],
                    "task_subtask_pair": data[3],
                    "instruction": data[4].item(),
                }

                list_dict_data.append(dict_data)

            dataset = datasets.Dataset.from_list(list_dict_data)
            # save datset
            dataset.save_to_disk(
                f"{raw_data_root}/{task_name}_subtask-{subtask_idx}_{split}"
            )


    trainsets = []
    testsets = []
    valsets = []
    trainsets_dict = {}
    testsets_dict = {}
    valsets_dict = {}

    for task_subtask_pair in task_subtask_pairs:
        task, subtask_idx = task_subtask_pair
        trainset = datasets.Dataset.load_from_disk(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_train"
        )
        trainsets.append(trainset)
        trainsets_dict[task_subtask_pair] = trainset
        valset = datasets.Dataset.load_from_disk(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_val"
        )
        valsets.append(valset)
        valsets_dict[task_subtask_pair] = valset
        testset = datasets.Dataset.load_from_disk(
            f"{raw_data_root}/{task}_subtask-{subtask_idx}_test"
        )
        testsets.append(testset)
        testsets_dict[task_subtask_pair] = testset

        print(f"{task}_{subtask_idx} loaded")

    concat_trainset = datasets.concatenate_datasets(trainsets)
    concat_testset = datasets.concatenate_datasets(testsets)
    #concat_testset = datasets.concatenate_datasets(testsets + trainsets + valsets)
        
    from transformers import AutoTokenizer
    system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."


    def prepare_data_instance(
            data_instance,
            system_prompt,
            mol_token="<mol>",
            num_query_tokens=32,
    ):
        input_mol_string = data_instance["input_mol_string"]
        input_mol_string = input_mol_string.replace("<SELFIES>", "<SELFIES> ").replace("</SELFIES>", " </SELFIES>")
        input_prompt = data_instance["instruction"]


        graph_sequence = "<GRAPH>" + mol_token * num_query_tokens + "</GRAPH>"
        input_mol_string += graph_sequence
        assert "<INPUT>" in input_prompt, f"llm_prompt should contain <INPUT_MOL>"

        input_prompt = input_prompt.replace("<INPUT>", input_mol_string)

        formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [INST]"
        formatted_target_text = data_instance["label"] + " </s>"

        if "additional" in data_instance["task_subtask_pair"]:
            convert_dict = {
                'qm9_additional_label/mu' : "qm9_dipole_moment",
                'qm9_additional_label/alpha' : "qm9_isotropic_polarizability",
                'qm9_additional_label/r2' : "qm9_electronic_spatial_extent",
                'qm9_additional_label/zpve' : "qm9_zero_point_vibrational_energy",

            }
            task = convert_dict[data_instance["task_subtask_pair"]]
        else:
            task = data_instance["task_subtask_pair"]

        data ={
            "task": task,
            "x": data_instance["x"],
            "edge_index": data_instance["edge_index"],
            "edge_attr": data_instance["edge_attr"],
            "additional_x": data_instance["x"],
            "additional_edge_index": data_instance["edge_index"],
            "additional_edge_attr": data_instance["edge_attr"],
            "prompt_text": formatted_prompt_text,
            "target_text": formatted_target_text,
        }
        return data

    
    data_instance = concat_testset[0]

    remove_keys = set(concat_testset.column_names)
    remove_keys -= {
        "task",
        "x",
        "edge_index",
        "edge_attr",
        "additional_x",
        "additional_edge_index",
        "additional_edge_attr",
    }
    mapped_trainset = concat_trainset.map(
        lambda x: prepare_data_instance(
            x, system_prompt=system_prompt,
        ),
            # use 36 processes
            num_proc=args.train_procs
    )
    mapped_testset = concat_testset.map(
        lambda x: prepare_data_instance(
            x, system_prompt=system_prompt,
        ),
            # use 36 processes
            num_proc=args.test_procs
    )
    llm_model = "mistralai/Mistral-7B-Instruct-v0.3"
    mol_representation = "string+graph"
    num_query_token = 32
    base_model = llm_model.replace("/", "-")
    tags = [base_model, mol_representation]
    if "graph" in mol_representation:
        tags += [f"q{num_query_token}"]
    
    processed_file_name = "_".join(tags)

    mapped_trainset.save_to_disk(f"{raw_data_root}/{processed_file_name}_train_{cfg.data_tag}")
    mapped_testset.save_to_disk(f"{raw_data_root}/{processed_file_name}_test_{cfg.data_tag}")
    mapped_testset.save_to_disk(f"{raw_data_root}/{processed_file_name}_validation_{cfg.data_tag}")

    a = 17
