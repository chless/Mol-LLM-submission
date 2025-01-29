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


valence_dict = {
    "H": 1,
    "He": 0,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "As": 3,
    "Se": 2,
    "Br": 1,
    "Te": 2,
    "I": 1,
    # Transition metals and other elements can be added here
}


def substitute_atoms_based_on_selfies(selfies, min_r=0.3, max_r=0.9):
    edit_selfies = copy.copy(selfies)
    atoms = [
        atom
        for atom in re.findall("\[.+?\]", selfies)
        if "Ring" not in atom and "Branch" not in atom
    ]
    min_atoms = max(1, int(min_r * len(atoms)))
    max_atoms = min(int(max_r * len(atoms)), len(atoms) - 1)
    if min_atoms >= max_atoms:
        return edit_selfies + edit_selfies
    else:
        num_atoms_to_substitute = np.random.randint(min_atoms, max_atoms)
        while num_atoms_to_substitute > 0:
            selected_atom = np.random.choice(atoms).item()
            edit_selfies_parts = list(re.finditer("\[.+?\]", edit_selfies))
            edit_part = np.random.choice(edit_selfies_parts)
            new_selfies = (
                edit_selfies[: edit_part.start()]
                + selected_atom
                + edit_selfies[edit_part.end() :]
            )

            edit_selfies = new_selfies
            num_atoms_to_substitute -= 1

        return edit_selfies


def substitute_atoms_based_on_graph(graph, min_r=0.3, max_r=0.9):
    num_atoms = graph.x.size(0)
    min_atoms = max(1, int(min_r * num_atoms))
    max_atoms = int(max_r * num_atoms)

    edit_graph = graph.clone()

    if min_atoms >= max_atoms:
        print(f"min_atoms={min_atoms} >= max_atoms={max_atoms}")
        return edit_graph
    else:

        num_atoms_to_substitute = np.random.randint(min_atoms, max_atoms)

        original_indices = np.arange(num_atoms)
        shuffled_indices = shuffle_partial(
            original_indices.tolist(), num_to_shuffle=num_atoms_to_substitute
        )
        for i in range(len(shuffled_indices)):
            edit_graph.x[i] = graph.x[shuffled_indices[i]]

        return edit_graph


def remove_atoms_based_on_selfies(selfies, num_atoms_to_remove):
    edit_selfies = copy.copy(selfies)
    while num_atoms_to_remove > 0:
        try:
            edit_selfies_parts = list(re.finditer("\[.+?\]", edit_selfies))
            selected_part = np.random.choice(edit_selfies_parts)
            edit_index = selected_part.start()
            edit_len = len(selected_part.group())
            new_selfies = (
                edit_selfies[:edit_index] + edit_selfies[edit_index + edit_len :]
            )

            new_smiles = sf.decoder(edit_selfies)
            new_mol = Chem.MolFromSmiles(new_smiles)
            Chem.SanitizeMol(new_mol)

            edit_selfies = new_selfies
            num_atoms_to_remove -= 1
        except:
            break

    return edit_selfies


def prepare_rejected_mol(mol, preference_type="negative-size"):
    if preference_type == "negative-size":
        return augment_molecular_size_based_on_mol(mol)
    elif preference_type == "negative-structure":
        return augment_molecular_structure(mol)
    else:
        raise ValueError("preference_type should be one of 'size', 'structure'")


def augment_molecular_size_based_on_mol(mol, min_r=0.3, max_r=0.9):
    num_atoms = mol.GetNumAtoms()
    min_atoms = max(1, int(num_atoms * min_r))
    max_atoms = min(int(num_atoms * max_r), num_atoms - 1)

    if min_atoms >= max_atoms:
        edit_mol = add_atoms_based_on_mol(mol, num_atoms_to_add=min_atoms)
    else:
        num_changing_atoms = np.random.randint(min_atoms, max_atoms)

        prob = np.random.rand()

        if prob > 0.5:
            edit_mol = add_atoms_based_on_mol(mol, num_atoms_to_add=num_changing_atoms)
        else:
            edit_mol = remove_atoms_based_on_mol(
                mol, num_atoms_to_remove=num_changing_atoms
            )

    assert edit_mol is not None

    return edit_mol


def remove_atoms_based_on_mol(mol, num_atoms_to_remove):
    assert (
        num_atoms_to_remove < mol.GetNumAtoms()
    ), "num_atoms_to_remove should be less than the number of atoms in the molecule."
    assert num_atoms_to_remove > 0, "num_atoms_to_remove should be positive."

    sanitized_mol = Chem.RWMol(mol)

    while num_atoms_to_remove > 0:
        potential_remove_indices = []
        for atom in sanitized_mol.GetAtoms():
            # Check if all neighbors are connected by single bonds
            # to guarantee that the new bond will be single, so that might be kekulizable...
            all_single_bonds = True
            for neighbor in atom.GetNeighbors():
                bond = sanitized_mol.GetBondBetweenAtoms(
                    atom.GetIdx(), neighbor.GetIdx()
                )
                if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                    all_single_bonds = False
                    break
            if all_single_bonds:
                potential_remove_indices.append(atom.GetIdx())

        if not potential_remove_indices:
            break

        np.random.shuffle(potential_remove_indices)

        change_made = False
        for atom_index in potential_remove_indices:
            try:
                rw_mol = Chem.RWMol(sanitized_mol)
                rw_mol.RemoveAtom(atom_index)
                Chem.SanitizeMol(rw_mol)
                sanitized_mol = rw_mol.GetMol()
                num_atoms_to_remove -= 1
                change_made = True
                break
            except:
                continue

        if not change_made:
            break

    out = {
        "mol": sanitized_mol,
        "original_num_atoms": mol.GetNumAtoms(),
        "augmeted_num_atoms": sanitized_mol.GetNumAtoms(),
        "num_not_fulfilled_changes": num_atoms_to_remove,
    }
    return out


def add_atoms_based_on_mol(mol, num_atoms_to_add):
    assert num_atoms_to_add > 0, "num_atoms_to_add should be positive."

    sanitized_mol = Chem.RWMol(mol)
    atom_counts = get_unique_atoms_and_counts(sanitized_mol)

    while num_atoms_to_add > 0:
        potential_bond_indices = []
        for atom in sanitized_mol.GetAtoms():
            if atom.GetSymbol() not in valence_dict:
                continue
            if atom.GetExplicitValence() < valence_dict[atom.GetSymbol()]:
                # Check if all neighbors are connected by single bonds
                # to guarantee that the new bond will be single, so that might be kekulizable...
                all_single_bonds = True
                for neighbor in atom.GetNeighbors():
                    bond = sanitized_mol.GetBondBetweenAtoms(
                        atom.GetIdx(), neighbor.GetIdx()
                    )
                    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                        all_single_bonds = False
                        break
                if all_single_bonds:
                    potential_bond_indices.append(atom.GetIdx())

        if not potential_bond_indices:
            break

        np.random.shuffle(potential_bond_indices)

        change_made = False
        for atom_index in potential_bond_indices:
            try:
                rw_mol = Chem.RWMol(sanitized_mol)
                new_atom = Chem.Atom(sample_atom(atom_counts))
                new_atom_idx = rw_mol.AddAtom(new_atom)
                rw_mol.AddBond(
                    atom_index, new_atom_idx, order=Chem.rdchem.BondType.SINGLE
                )
                Chem.SanitizeMol(rw_mol)
                sanitized_mol = rw_mol.GetMol()
                atom_counts[new_atom.GetSymbol()] += 1
                num_atoms_to_add -= 1
                change_made = True
                break
            except:
                continue

        if not change_made:
            break

    out = {
        "mol": sanitized_mol,
        "original_num_atoms": mol.GetNumAtoms(),
        "augmeted_num_atoms": sanitized_mol.GetNumAtoms(),
        "num_not_fulfilled_changes": num_atoms_to_add,
    }
    return out


def get_unique_atoms_and_counts(molecule):
    """
    Get a list of unique atoms and their counts in the given RDKit molecule object.

    Args:
    molecule (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
    dict: A dictionary with atom symbols as keys and their counts as values.
    """
    # Extract atom symbols from the molecule
    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]

    # Count occurrences of each atom symbol
    atom_counts = Counter(atom_symbols)

    return dict(atom_counts)


def sample_atom(atom_counts):
    """
    Sample an atom from the atom_counts with a probability proportional to its count.

    Args:
    atom_counts (dict): Dictionary of atom symbols and their counts.

    Returns:
    str: A sampled atom symbol based on its relative abundance.
    """
    # Extract atoms and their respective counts
    atoms = list(atom_counts.keys())
    counts = list(atom_counts.values())

    # Sample one atom based on the counts as weights
    sampled_atom = np.random.choice(atoms, p=np.array(counts) / np.sum(counts)).item()

    return sampled_atom


def augment_molecular_structure(selfies: list[str], min_r=0.1, max_r=1.0):

    return


from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def mol2graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


def graph2data(graph):
    data = Data(
        x=torch.tensor(graph["node_feat"], dtype=torch.int64),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.int64),
        edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.int64),
    )
    return data


import random


def shuffle_partial(lst, num_to_shuffle=0):
    # Step 1: Calculate the number of elements to shuffle
    n = len(lst)

    # Step 2: Randomly select indices to shuffle
    indices_to_shuffle = random.sample(range(n), num_to_shuffle)

    # Step 3: Extract the elements at these indices
    elements_to_shuffle = [lst[i] for i in indices_to_shuffle]

    # Step 4: Shuffle the selected elements
    random.shuffle(elements_to_shuffle)

    # Step 5: Replace the original elements with the shuffled ones
    for i, idx in enumerate(indices_to_shuffle):
        lst[idx] = elements_to_shuffle[i]

    return lst


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

        self.apply_simpo = args.train_simpo if self.train else args.eval_simpo

        self.projector_type = args.projector_type
        if hasattr(args, "simpo_modality"):
            self.simpo_modality = args.simpo_modality
            assert self.simpo_modality == "graph"
        
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

        if self.apply_simpo:
            prompt_text = prompt_text + prompt_text
            target_text = target_text + target_text
            tasks = tasks + tasks

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

            if self.apply_simpo:
                if "graph" in self.simpo_modality:
                    list_rejected_graphs = [
                        Data(
                            x=torch.tensor(sample["rejected_x"], dtype=torch.int64),
                            edge_index=torch.tensor(sample["rejected_edge_index"], dtype=torch.int64),
                            edge_attr=torch.tensor(sample["rejected_edge_attr"], dtype=torch.int64),
                        )
                        for sample in batch
                    ]
                    # for reagent prediction
                    list_rejected_additional_graphs = [
                        Data(
                            x=torch.tensor(sample["additional_rejected_x"], dtype=torch.int64),
                            edge_index=torch.tensor(
                                sample["additional_rejected_edge_index"], dtype=torch.int64
                            ),
                            edge_attr=torch.tensor(
                                sample["additional_rejected_edge_attr"], dtype=torch.int64
                            ),
                        )
                        for sample in batch
                    ]
                else:
                    list_rejected_graphs = copy.deepcopy(list_graphs)
                    list_rejected_additional_graphs = copy.deepcopy(
                        list_additional_graphs
                    )

                list_graphs = list_graphs + list_rejected_graphs
                list_additional_graphs = (
                    list_additional_graphs + list_rejected_additional_graphs
                )

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


def generate_text(data_point, tokenizer, args, test=False):
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

    input_mol_string = "None"
    if input_mol_string_pattern.search(input_text):
        input_mol_string = input_mol_string_pattern.search(input_text).group()

    prompt_text = tokenizer.bos_token + input_text
    target_text = output_text + " " + tokenizer.eos_token

    text_result = {
        "prompt_text": prompt_text,
        "target_text": target_text,
        "input_mol_string": input_mol_string,  # TODO: later, save rdkit mol object
    }

    return text_result


def random_replace_mol_string(input_tokens_input_ids, tokenizer):
    ids = input_tokens_input_ids
    mol_string_randomization_ratio = tokenizer.mol_string_randomization_ratio
    total_selfies_token_ids = tokenizer.selfies_token_ids

    selfies_min_id = min(total_selfies_token_ids)
    selfies_max_id = max(total_selfies_token_ids)
    # if ids are correspond to total_selfies_token_ids, replace them with random token by mol_string_randomization_ratio
    full_random_replaced = torch.where(
        (ids >= selfies_min_id) & (ids <= selfies_max_id),
        torch.randint(selfies_min_id, selfies_max_id + 1, ids.shape, device=ids.device),
        ids,
    )
    partial_random_replaced = torch.where(
        torch.rand(ids.shape, device=ids.device) < mol_string_randomization_ratio,
        full_random_replaced,
        ids,
    )
    return partial_random_replaced
