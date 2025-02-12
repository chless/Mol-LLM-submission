from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
import numpy as np
from tqdm import tqdm
import instructions_smol
import datasets
from datasets import load_dataset
import pandas as pd
import os
from rdkit import Chem


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
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
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
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 

from rdkit import Chem
import selfies as sf
from download_dataset import wrap_label

system_prompt = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation."

def prepare_data_instance(
        mol,
        label,
        task,
        instruction_templates,
        system_prompt,
        mol_token="<mol>",
        num_query_tokens=32,
):

    label = wrap_label(label, task=task)
    input_prompt = np.random.choice(instruction_templates).item()
    assert "<INPUT>" in input_prompt, f"llm_prompt should contain <INPUT>"
    graph_sequence = "<GRAPH>" + mol_token * num_query_tokens + "</GRAPH>"

    if "reagent_prediction" in task:
        smiles = Chem.MolToSmiles(mol[0])
        selfies = sf.encoder(smiles)
        input_mol_string = "<SELFIES> " + selfies + " </SELFIES>"
        input_mol_string_graph = input_mol_string + graph_sequence

        additional_smiles = Chem.MolToSmiles(mol[1])
        additional_selfies = sf.encoder(additional_smiles)
        additional_input_mol_string = "<SELFIES> " + additional_selfies + " </SELFIES>"
        additional_input_mol_string_graph = additional_input_mol_string + graph_sequence

        input_mol_string = input_mol_string + "|>>|" + additional_input_mol_string
        input_mol_string_graph = input_mol_string_graph + "|>>|" + additional_input_mol_string_graph
        
        graph = mol2graph(mol[0])
        additional_graph = mol2graph(mol[1])
    else:
        assert isinstance(mol, Chem.Mol), f"mol should be a RDKit Mol object, but got {type(mol)}"
        mol = mol
        smiles = Chem.MolToSmiles(mol)
        selfies = sf.encoder(smiles)
        input_mol_string = "<SELFIES> " + selfies + " </SELFIES>"
        input_mol_string_graph = input_mol_string + graph_sequence

        graph = mol2graph(mol)
        additional_graph = graph
    
    input_prompt = input_prompt.replace("<INPUT>", input_mol_string_graph)

    formatted_prompt_text = "<s>[INST] " + system_prompt + " \n\n" + input_prompt + " [/INST] "
    formatted_target_text = label + " </s>"


    data = {
        "task": task,
        "x": graph['node_feat'],
        "edge_index": graph['edge_index'],
        "edge_attr": graph['edge_feat'],
        "additional_x": additional_graph['node_feat'],
        "additional_edge_index": additional_graph['edge_index'],
        "additional_edge_attr": additional_graph['edge_feat'],
        "input_mol_string": input_mol_string,
        "prompt_text": formatted_prompt_text,
        "target_text": formatted_target_text,
    }
    return data


def get_data_list(
        list_mol, list_label, task, instruction_templates, system_prompt
):
    list_data = []
    iter_bar = tqdm(range(len(list_mol)))

    for i in iter_bar:
        data = prepare_data_instance(
        mol=list_mol[i],
        label=list_label[i], 
        task=task,
        instruction_templates=instruction_templates,
        system_prompt=system_prompt
    )  
        list_data.append(data)
    return list_data
