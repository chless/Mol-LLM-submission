import torch
from transformers import DataCollatorForSeq2Seq
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater

import numpy as np

from collections import Counter

import selfies as sf

import rdkit.Chem as Chem
import re
import copy

CLASSIFICATION_BENCHMARKS = [
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",
]
REGRESSION_BENCHMARKS = [
    "smol-property_prediction-esol",
    "smol-property_prediction-lipo",
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
    "qm9_dipole_moment",
    "qm9_isotropic_polarizability",
    "qm9_electronic_spatial_extent",
    "qm9_zero_point_vibrational_energy",
    "qm9_heat_capacity_298K",
    "qm9_internal_energy_298K",
    "qm9_enthalpy_298K",
    "qm9_free_energy_298K",
    "alchemy_homo",
    "alchemy_lumo",
    "alchemy_homo_lumo_gap",
]
REACTION_BENCHMARKS = [
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
    "presto-forward_reaction_prediction",
    "presto-retrosynthesis",
    "presto-reagent_prediction",
]
TEXT2MOL_BENCHMARKS = [
    "chebi-20-text2mol",
    "smol-molecule_generation",
]
MOL2TEXT_BENCHMARKS = [
    "chebi-20-mol2text",
    "smol-molecule_captioning",
]
NAME_CONVERSION_BENCHMARKS = [
    "smol-name_conversion-i2s",
    "smol-name_conversion-i2f",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
]


tasks = (
    CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
    + REACTION_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + NAME_CONVERSION_BENCHMARKS
)

input_mol_string_pattern = re.compile("<SELFIES>.*?</SELFIES>")
graph_sequence = re.compile("<GRAPH>[<mol>]+?</GRAPH>")


def task2id(task):
    # task name to task id
    task2id = {k: i for i, k in enumerate(tasks)}
    return task2id[task]


def id2task(task_id):
    # task id to task name
    id2task = {i: k for i, k in enumerate(tasks)}
    return id2task[task_id]


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors=None,
        train=True,
        args=None,
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.train = train
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        self.mol_representation = args.mol_representation

        self.apply_molpo = args.train_simpo if self.train else args.eval_simpo

        self.projector_type = args.projector_type

        
        if self.mol_representation in ["string+graph", "graph_only"]:
            self.graph_collator = GraphCollater([], [])

    def select_mol_representation(self, prompt_text, mol_representation="string+graph"):
        if mol_representation == "string+graph":
            return prompt_text
        elif mol_representation == "string_only":
            string_only_prompt_text = [graph_sequence.sub("", p) for p in prompt_text]
            return string_only_prompt_text
        elif mol_representation == "graph_only":
            graph_only_prompt_text = [
                input_mol_string_pattern.sub("", p) for p in prompt_text
            ]
            return graph_only_prompt_text
        else:
            raise ValueError(
                "mol_representation should be one of ['string+graph', 'string_only', 'graph_only']"
            )

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        tasks = [task2id(sample.pop("task")) for sample in batch]  # task id
        prompt_text = [sample["prompt_text"] for sample in batch]
        target_text = [sample["target_text"] for sample in batch]

        prompt_text = self.select_mol_representation(
            prompt_text, mol_representation=self.mol_representation
        )

        if self.apply_molpo:
            if self.train:
                self.reject_cardinal = self.current_epoch
            else:
                self.reject_cardinal = 0

            # prepare tuples
            # sft tuple (gw, sw, q, y)
            # molpo chosen tuple (gw, sl, q, y)
            # molpo rejected tuple (gl, sl, q, y)

            prompt_text_sl = []
            for i in range(len(prompt_text)):
                sw = batch[i]["input_mol_string"].replace('<SELFIES>', "").replace('</SELFIES>', "").replace(' ', '')
                sl = random_noise_selfies(selfies=sw, tokenizer=self.tokenizer)
                prompt_sl = input_mol_string_pattern.sub(
                    f"<SELFIES> {sl} </SELFIES>", prompt_text[i]
                )
                prompt_text_sl.append(prompt_sl)

            prompt_text = prompt_text + prompt_text_sl * 2 # ((q, sw), (q, sl), (q, sl))
            target_text = target_text * 3 # (y, y, y)
            tasks = tasks * 3

        if "graph" in self.mol_representation:
            list_graphs = [
                Data(
                    x=torch.tensor(sample["x"], dtype=torch.int64),
                    edge_index=torch.tensor(sample["edge_index"], dtype=torch.int64),
                    edge_attr=torch.tensor(sample["edge_attr"], dtype=torch.int64),
                )
                for sample in batch
            ]
            # for reagent prediction
            list_additional_graphs = [
                Data(
                    x=torch.tensor(sample["additional_x"], dtype=torch.int64),
                    edge_index=torch.tensor(
                        sample["additional_edge_index"], dtype=torch.int64
                    ),
                    edge_attr=torch.tensor(
                        sample["additional_edge_attr"], dtype=torch.int64
                    ),
                )
                for sample in batch
            ]

            if self.apply_molpo:
                list_rejected_graphs = [
                Data(
                    x=torch.tensor(sample[f"{self.reject_cardinal}-th_rejected_x"], dtype=torch.int64),
                    edge_index=torch.tensor(sample[f"{self.reject_cardinal}-th_rejected_edge_index"], dtype=torch.int64),
                    edge_attr=torch.tensor(sample[f"{self.reject_cardinal}-th_rejected_edge_attr"], dtype=torch.int64),
                )
                for sample in batch
                ]
                # for reagent prediction
                list_rejected_additional_graphs = [
                    Data(
                        x=torch.tensor(sample[f"{self.reject_cardinal}-th_additional_rejected_x"], dtype=torch.int64),
                        edge_index=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_additional_rejected_edge_index"], dtype=torch.int64
                        ),
                        edge_attr=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_additional_rejected_edge_attr"], dtype=torch.int64
                        ),
                    )
                    for sample in batch
                ]

                # (gw, gw, gl)
                list_graphs = list_graphs * 2 + list_rejected_graphs
                list_additional_graphs = list_additional_graphs * 2 + list_rejected_additional_graphs

        if self.projector_type == "mlp" and "graph" in self.mol_representation:
            # TODO: implement for reagent prediction
            for i in range(len(prompt_text)):
                num_nodes_in_graph = list_graphs[i].x.size(0)
                num_nodes_mol = "<mol>" * num_nodes_in_graph
                mol_tokens_pattern = re.compile("(<mol>)+")
                prompt_text[i] = mol_tokens_pattern.sub(num_nodes_mol, prompt_text[i])

        self.tokenizer.padding_side = "left"
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        target_tokenized = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        full_input_ids = [
            p + t
            for p, t in zip(
                prompt_tokenized["input_ids"], target_tokenized["input_ids"]
            )
        ]
        full_attention_mask = [
            p + t
            for p, t in zip(
                prompt_tokenized["attention_mask"], target_tokenized["attention_mask"]
            )
            ]

        prompt_length = [len(p) for p in prompt_tokenized["input_ids"]]
        full_input_ids = [f_ids[: self.max_length] for f_ids in full_input_ids]
        full_attention_mask = [
            f_ids[: self.max_length] for f_ids in full_attention_mask
        ]

        self.tokenizer.padding_side = "left"
        features = self.tokenizer.pad(
            {"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if not self.train:
            prompt_features = self.tokenizer.pad(
                {
                    "input_ids": [p for p in prompt_tokenized["input_ids"]],
                    "attention_mask": [p for p in prompt_tokenized["attention_mask"]],
                },
                padding=self.padding ,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features["prompt_input_ids"] = prompt_features.input_ids  # ['input_ids']
            features["prompt_attention_mask"] = (
                prompt_features.attention_mask
            )  # ['attention_mask']

            self.tokenizer.padding_side = "right"
            eval_features = self.tokenizer.pad(
                {
                    "input_ids": [t for t in target_tokenized["input_ids"]],
                },
                padding=self.padding ,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            eval_features.input_ids = eval_features.input_ids.masked_fill(
                eval_features.input_ids == self.tokenizer.pad_token_id, -100
            )
            features["eval_labels"] = eval_features.input_ids
            eval_simpo_labels = eval_features.input_ids.clone()
            for simpo_mask_id in self.tokenizer.simpo_mask_ids:
                eval_simpo_labels = eval_simpo_labels.masked_fill(eval_simpo_labels == simpo_mask_id, -100)
            features["eval_simpo_labels"] = eval_simpo_labels

        labels_ids = torch.full_like(features["input_ids"], self.tokenizer.pad_token_id)
        for i, target in enumerate(target_tokenized["input_ids"]):
            label = target
            if prompt_length[i] >= self.max_length:
                continue
            else:
                len_label = min(len(label), self.max_length - prompt_length[i])
                labels_ids[i, -len_label:] = torch.tensor(
                    label[:len_label], dtype=torch.int64
                )

        labels_ids = labels_ids.masked_fill(
            labels_ids == self.tokenizer.pad_token_id, -100
        )
        features["labels"] = labels_ids
        simpo_labels_ids = labels_ids.clone()
        for simpo_mask_id in self.tokenizer.simpo_mask_ids:
            simpo_labels_ids = simpo_labels_ids.masked_fill(simpo_labels_ids == simpo_mask_id, -100)
        features["simpo_labels"] = simpo_labels_ids

        assert (
            features.input_ids.size(1) <= self.max_length
        ), f"features.input_ids.size(1)={features.input_ids.size(1)} > self.max_length={self.max_length}"
        assert (
            features.labels.size(1) <= self.max_length
        ), f"features.labels.size(1)={features.labels.size(1)} > self.max_length={self.max_length}"

        features["tasks"] = torch.tensor(tasks, dtype=torch.int16)
        if "graph" in self.mol_representation:
            graphs = self.graph_collator(list_graphs)
            additional_graphs = self.graph_collator(list_additional_graphs)
            features["graphs"] = graphs
            features["additional_graphs"] = additional_graphs
            features["is_mol_token"] = (
                features["input_ids"] == self.tokenizer.mol_token_id
            )
            if not self.train:
                features["prompt_is_mol_token"] = (
                    torch.tensor(features["prompt_input_ids"])
                    == self.tokenizer.mol_token_id
                )

        return features



def random_noise_selfies(selfies, tokenizer, noise_ratio=0.3):
    selfies_ids = tokenizer.encode(selfies, add_special_tokens=False)
    total_selfies_token_ids = tokenizer.selfies_token_ids
    num_ids_to_replace = int(noise_ratio * len(selfies_ids))
    replacing_random_ids = np.random.choice(
        total_selfies_token_ids, num_ids_to_replace, replace=True
    )

    # replace selfies_ids with randomly selected total_selfies_token_ids as many as num_ids_to_replace
    position_to_replace = np.random.choice(
        len(selfies_ids), num_ids_to_replace, replace=False
    )
    noised_selfies_ids = copy.deepcopy(selfies_ids)
    for i, replance_idx in enumerate(position_to_replace):
        noised_selfies_ids[replance_idx] = replacing_random_ids[i]


    noised_selfies = tokenizer.decode(noised_selfies_ids, skip_special_tokens=True).replace(' ', '')
    return noised_selfies
