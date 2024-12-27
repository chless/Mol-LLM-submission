import torch
from transformers import DataCollatorForSeq2Seq
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater

import numpy as np

import re


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
]
REACTION_BENCHMARKS = [
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
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

input_mol_string_pattern = re.compile("<SELFIES>" + ".*?" + "</SELFIES>")


def task2id(task):
    # task name to task id
    task2id = {k: i for i, k in enumerate(tasks)}
    return task2id[task]


def id2task(task_id):
    # task id to task name
    id2task = {i: k for i, k in enumerate(tasks)}
    return id2task[task_id]


import selfies as sf

from ogb.utils import smiles2graph

import rdkit.Chem as Chem


def get_selfies_from_input_ids(input_ids, tokenizer):
    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    selfies = []
    for text in input_texts:
        # TODO: implement for reagent prediction, which gives two molecule
        searched_selfies = input_mol_string_pattern.search(text).group()
        searched_selfies = (
            searched_selfies.replace("<SELFIES>", "")
            .replace("</SELFIES>", "")
            .replace(" ", "")
        )
        selfies.append(searched_selfies)

    return selfies


import random


def inject_random_C_sequence(selfies: list[str], min_r=0.1, max_r=1.0):
    # inject random C atoms to the selfies
    # ratio: the ratio of the number of C atoms to the number of atoms in the molecule
    mol = Chem.MolFromSmiles(sf.decoder(selfies))
    num_atoms = mol.GetNumAtoms()
    # count number of C
    # TODO: reducing C case learning
    num_existing_C = mol.GetNumAtoms(6)
    min_C_atoms = int(num_atoms * min_r)
    max_C_atoms = int(num_atoms * max_r)
    num_C_atoms = np.random.randint(min_C_atoms, max_C_atoms) + 1
    # convert to smiles
    C_smiles = "C" * num_C_atoms
    C_selfies = sf.encoder(C_smiles)
    # replace random '[C]' in the selfies with C_selfies
    pattern = r"\[C\]"
    matches = list(re.finditer(pattern, selfies))
    num_matches = len(matches)

    if num_matches > 0:
        random_index = random.randint(0, num_matches - 1)

        # Get the start position of the match
        match_start = matches[random_index].start()

        # Modify the string
        before = selfies[:match_start]  # Include the ']'
        after = selfies[match_start:]  # The rest after ']'
        after = after.replace("[C]", C_selfies, 1)  # Replace first '[' with 'None'

        injected_selfies = before + after
    return injected_selfies


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph["node_feat"])
    edge_index = torch.from_numpy(
        graph["edge_index"],
    )
    edge_attr = torch.from_numpy(graph["edge_feat"])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def convert_selfies_to_graph(selfies: list[str]):
    smiles = sf.decoder(selfies)
    graph = smiles2data(smiles)
    return graph


def prepare_rejected_graphs(selfies: list[str]):
    rejected_graphs = []
    for i in selfies:
        rejected_selfies = inject_random_C_sequence(i)
        rejected_graph = convert_selfies_to_graph(rejected_selfies)
        rejected_graphs.append(rejected_graph)
    return rejected_graphs


def prepare_rejected_selfies(selfies: list[str]):
    rejected_selfies = []

    return


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors=None,
        use_graph=False,
        train=True,
        mdpo=False,
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.use_graph = use_graph
        self.train = train
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        self.mdpo = mdpo

        if self.use_graph:
            # Collater with no special follow_batch or exclude_keys
            self.graph_collator = GraphCollater([], [])

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        tasks = [task2id(sample.pop("task")) for sample in batch]  # task id

        prompt_tokenized = [sample["prompt_tokenized"] for sample in batch]
        target_tokenized = [sample["target_tokenized"] for sample in batch]

        if self.use_graph:
            graphs = [
                Data(
                    x=torch.tensor(sample["x"], dtype=torch.int64),
                    edge_index=torch.tensor(sample["edge_index"], dtype=torch.int64),
                    edge_attr=torch.tensor(sample["edge_attr"], dtype=torch.int64),
                )
                for sample in batch
            ]
            # for reagent prediction
            additional_graphs = [
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

            if self.mdpo:
                list_selfies = get_selfies_from_input_ids(
                    [p["input_ids"] for p in prompt_tokenized], self.tokenizer
                )
                rejected_graphs = prepare_rejected_graphs(list_selfies)
                rejected_additional_graphs = prepare_rejected_graphs(list_selfies)

        full_input_ids = [
            p["input_ids"] + t["input_ids"]
            for p, t in zip(prompt_tokenized, target_tokenized)
        ]
        full_attention_mask = [
            p["attention_mask"] + t["attention_mask"]
            for p, t in zip(prompt_tokenized, target_tokenized)
        ]

        prompt_length = [len(p["input_ids"]) for p in prompt_tokenized]
        full_input_ids = [f_ids[: self.max_length] for f_ids in full_input_ids]
        full_attention_mask = [
            f_ids[: self.max_length] for f_ids in full_attention_mask
        ]

        features = self.tokenizer.pad(
            {"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if not self.train:
            prompt_features = self.tokenizer.pad(
                {
                    "input_ids": [p["input_ids"] for p in prompt_tokenized],
                    "attention_mask": [p["attention_mask"] for p in prompt_tokenized],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features["prompt_input_ids"] = prompt_features.input_ids  # ['input_ids']
            features["prompt_attention_mask"] = (
                prompt_features.attention_mask
            )  # ['attention_mask']

        if self.tokenizer.padding_side == "right":
            raise NotImplementedError("padding_side should be left")

        labels_ids = torch.full_like(features["input_ids"], self.tokenizer.pad_token_id)
        for i, target in enumerate(target_tokenized):
            label = target["input_ids"]
            if prompt_length[i] >= self.max_length:
                continue
            else:
                len_label = min(len(label), self.max_length - prompt_length[i])
                labels_ids[i, -len_label:] = torch.tensor(
                    label[:len_label], dtype=torch.int64
                )

        features["labels"] = labels_ids

        assert (
            features.input_ids.size(1) <= self.max_length
        ), f"features.input_ids.size(1)={features.input_ids.size(1)} > self.max_length={self.max_length}"
        assert (
            features.labels.size(1) <= self.max_length
        ), f"features.labels.size(1)={features.labels.size(1)} > self.max_length={self.max_length}"

        features["tasks"] = torch.tensor(tasks, dtype=torch.int16)
        if self.use_graph:
            graphs = self.graph_collator(graphs)
            additional_graphs = self.graph_collator(additional_graphs)
            features["graphs"] = graphs
            features["additional_graphs"] = additional_graphs

            if self.mdpo:
                rejected_graphs = self.graph_collator(rejected_graphs)
                rejected_additional_graphs = self.graph_collator(
                    rejected_additional_graphs
                )
                features["rejected_graphs"] = rejected_graphs
                features["rejected_additional_graphs"] = rejected_additional_graphs

            features["is_mol_token"] = (
                torch.tensor(features["input_ids"]) == self.tokenizer.mol_token_id
            )
            if not self.train:
                features["prompt_is_mol_token"] = (
                    torch.tensor(features["prompt_input_ids"])
                    == self.tokenizer.mol_token_id
                )

        return features


def generate_and_tokenize_prompt(data_point, tokenizer, args, test=False):
    default_system_prompt = "You are a helpful assistant for molecular chemistry, \
to address tasks including molecular property classification, \
molecular property regression, chemical reaction prediction, \
molecule captioning, molecule generation. \n\n"

    input_text = "[INST] " + default_system_prompt + data_point["input"] + " [/INST] "
    output_text = data_point["output"]

    def postfix_graph_sequence(match):
        return match.group(0) + graph_sequence

    graph_sequence = "<GRAPH>" + "<mol>" * args.num_query_token + "</GRAPH>"

    if args.mol_representation == "graph_only":
        input_text = input_mol_string_pattern.sub(graph_sequence, input_text)
    elif args.mol_representation == "string+graph":
        input_text = input_mol_string_pattern.sub(postfix_graph_sequence, input_text)

    prompt_tokenized = tokenizer(
        tokenizer.bos_token + input_text,
        truncation=True,
        max_length=args.max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    target_tokenized = tokenizer(
        output_text + " " + tokenizer.eos_token,
        truncation=True,
        max_length=args.max_length - len(prompt_tokenized["input_ids"]),
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    tokenized_result = {
        "prompt_tokenized": prompt_tokenized,
        "target_tokenized": target_tokenized,
    }

    return tokenized_result
