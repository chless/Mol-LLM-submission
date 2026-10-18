from datasets import load_from_disk
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import re
import random
from ogb.utils.features import (
    allowable_features,
    atom_to_feature_vector,
    bond_to_feature_vector,
)
import selfies as sf
from tqdm import tqdm
from collections import Counter
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
)

smartsPatts = {
    2: ("[#104]", 0),
    3: ("[#32,#33,#34,#50,#51,#52,#82,#83,#84]", 0),
    4: ("[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]", 0),
    5: ("[Sc,Ti,Y,Zr,Hf]", 0),
    6: ("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]", 0),
    7: ("[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]", 0),
    8: ("[!#6;!#1]1~*~*~*~1", 0),
    9: ("[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]", 0),
    10: ("[Be,Mg,Ca,Sr,Ba,Ra]", 0),
    11: ("*1~*~*~*~1", 0),
    12: ("[Cu,Zn,Ag,Cd,Au,Hg]", 0),
    13: ("[#8]~[#7](~[#6])~[#6]", 0),
    14: ("[#16]-[#16]", 0),
    15: ("[#8]~[#6](~[#8])~[#8]", 0),
    16: ("[!#6;!#1]1~*~*~1", 0),
    17: ("[#6]#[#6]", 0),
    18: ("[#5,#13,#31,#49,#81]", 0),
    19: ("*1~*~*~*~*~*~*~1", 0),
    20: ("[#14]", 0),
    21: ("[#6]=[#6](~[!#6;!#1])~[!#6;!#1]", 0),
    22: ("*1~*~*~1", 0),
    23: ("[#7]~[#6](~[#8])~[#8]", 0),
    24: ("[#7]-[#8]", 0),
    25: ("[#7]~[#6](~[#7])~[#7]", 0),
    26: ("[#6]=;@[#6](@*)@*", 0),
    27: ("[I]", 0),
    28: ("[!#6;!#1]~[CH2]~[!#6;!#1]", 0),
    29: ("[#15]", 0),
    30: ("[#6]~[!#6;!#1](~[#6])(~[#6])~*", 0),
    31: ("[!#6;!#1]~[F,Cl,Br,I]", 0),
    32: ("[#6]~[#16]~[#7]", 0),
    33: ("[#7]~[#16]", 0),
    34: ("[CH2]=*", 0),
    35: ("[Li,Na,K,Rb,Cs,Fr]", 0),
    36: ("[#16R]", 0),
    37: ("[#7]~[#6](~[#8])~[#7]", 0),
    38: ("[#7]~[#6](~[#6])~[#7]", 0),
    39: ("[#8]~[#16](~[#8])~[#8]", 0),
    40: ("[#16]-[#8]", 0),
    41: ("[#6]#[#7]", 0),
    43: ("[!#6;!#1;!H0]~*~[!#6;!#1;!H0]", 0),
    45: ("[#6]=[#6]~[#7]", 0),
    47: ("[#16]~*~[#7]", 0),
    48: ("[#8]~[!#6;!#1](~[#8])(~[#8])", 0),
    50: ("[#6]=[#6](~[#6])~[#6]", 0),
    51: ("[#6]~[#16]~[#8]", 0),
    52: ("[#7]~[#7]", 0),
    53: ("[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]", 0),
    54: ("[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]", 0),
    55: ("[#8]~[#16]~[#8]", 0),
    56: ("[#8]~[#7](~[#8])~[#6]", 0),
    57: ("[#8R]", 0),
    58: ("[!#6;!#1]~[#16]~[!#6;!#1]", 0),
    60: ("[#16]=[#8]", 0),
    61: ("*~[#16](~*)~*", 0),
    62: ("*@*!@*@*", 0),
    63: ("[#7]=[#8]", 0),
    64: ("*@*!@[#16]", 0),
    66: ("[#6]~[#6](~[#6])(~[#6])~*", 0),
    67: ("[!#6;!#1]~[#16]", 0),
    68: ("[!#6;!#1;!H0]~[!#6;!#1;!H0]", 0),
    69: ("[!#6;!#1]~[!#6;!#1;!H0]", 0),
    70: ("[!#6;!#1]~[#7]~[!#6;!#1]", 0),
    71: ("[#7]~[#8]", 0),
    72: ("[#8]~*~*~[#8]", 0),
    73: ("[#16]=*", 0),
    74: ("[CH3]~*~[CH3]", 0),
    75: ("*!@[#7]@*", 0),
    76: ("[#6]=[#6](~*)~*", 0),
    77: ("[#7]~*~[#7]", 0),
    78: ("[#6]=[#7]", 0),
    79: ("[#7]~*~*~[#7]", 0),
    80: ("[#7]~*~*~*~[#7]", 0),
    81: ("[#16]~*(~*)~*", 0),
    82: ("*~[CH2]~[!#6;!#1;!H0]", 0),
    83: ("[!#6;!#1]1~*~*~*~*~1", 0),
    84: ("[NH2]", 0),
    85: ("[#6]~[#7](~[#6])~[#6]", 0),
    86: ("[C;H2,H3][!#6;!#1][C;H2,H3]", 0),
    87: ("[F,Cl,Br,I]!@*@*", 0),
    88: ("[#16]", 0),
    89: ("[#8]~*~*~*~[#8]", 0),
    92: ("[#8]~[#6](~[#7])~[#6]", 0),
    93: ("[!#6;!#1]~[CH3]", 0),
    94: ("[!#6;!#1]~[#7]", 0),
    95: ("[#7]~*~*~[#8]", 0),
    96: ("*1~*~*~*~*~1", 0),
    97: ("[#7]~*~*~*~[#8]", 0),
    98: ("[!#6;!#1]1~*~*~*~*~*~1", 0),
    99: ("[#6]=[#6]", 0),
    100: ("*~[CH2]~[#7]", 0),
    102: ("[!#6;!#1]~[#8]", 0),
    104: ("[!#6;!#1;!H0]~*~[CH2]~*", 0),
    105: ("*@*(@*)@*", 0),
    106: ("[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]", 0),
    107: ("[F,Cl,Br,I]~*(~*)~*", 0),
    108: ("[CH3]~*~*~*~[CH2]~*", 0),
    109: ("*~[CH2]~[#8]", 0),
    110: ("[#7]~[#6]~[#8]", 0),
    111: ("[#7]~*~[CH2]~*", 0),
    112: ("*~*(~*)(~*)~*", 0),
    114: ("[CH3]~[CH2]~*", 0),
    115: ("[CH3]~*~[CH2]~*", 0),
    117: ("[#7]~*~[#8]", 0),
    119: ("[#7]=*", 0),
    122: ("*~[#7](~*)~*", 0),
    123: ("[#8]~[#6]~[#8]", 0),
    124: ("[!#6;!#1]~[!#6;!#1]", 0),
    126: ("*!@[#8]!@*", 0),
    127: ("*@*!@[#8]", 1),
    130: ("[!#6;!#1]~[!#6;!#1]", 1),
    131: ("[!#6;!#1;!H0]", 1),
    132: ("[#8]~*~[CH2]~*", 0),
    133: ("*@*!@[#7]", 0),
    134: ("[F,Cl,Br,I]", 0),
    136: ("[#8]=*", 1),
    138: ("[!#6;!#1]~[CH2]~*", 1),
    139: ("[O;!H0]", 0),
    140: ("[#8]", 3),
    141: ("[CH3]", 2),
    142: ("[#7]", 1),
    143: ("*@*!@[#8]", 0),
    145: ("*1~*~*~*~*~*~1", 1),
    146: ("[#8]", 2),
    148: ("*~[!#6;!#1](~*)~*", 0),
    149: ("[C;H3,H4]", 1),
    150: ("*!@*@*!@*", 0),
    151: ("[#7;!H0]", 0),
    152: ("[#8]~[#6](~[#6])~[#6]", 0),
    153: ("[!#6;!#1]~[CH2]~*", 0),
    154: ("[#6]=[#8]", 0),
    155: ("*!@[CH2]!@*", 0),
    156: ("[#7]~*(~*)~*", 0),
    157: ("[#6]-[#8]", 0),
    158: ("[#6]-[#7]", 0),
    159: ("[#8]", 1),
    160: ("[C;H3,H4]", 0),
    161: ("[#7]", 0),
    163: ("*1~*~*~*~*~*~1", 0),
    164: ("[#8]", 0),
}

atom_groups = {
    "!#6;!#1": [
        "#2",
        "#3",
        "#4",
        "#5",
        "#7",
        "#8",
        "#10",
    ],
    "!#6;!#1;!H0": [
        "#2",
        "#3",
        "#4",
        "#5",
        "#7",
        "#8",
        "#10",
    ],
    "*": ["#6"],
    "~": ["-"],
    "R": ["#6"],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        safe_index(
            allowable_features["possible_chirality_list"], str(atom.GetChiralTag())
        ),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def mol2graph(mol):
    """
    Converts mol to graph Data object.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        dict: Graph representation with edge_index, edge_feat, node_feat, and num_nodes.
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    num_bond_features = 3
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T

        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


def graph2mol(graph):
    """
    Converts a graph representation of a molecule to an RDKit molecule.

    Args:
        graph (dict): A graph representation with 'edge_index', 'edge_feat', 'node_feat', and 'num_nodes'.

    Returns:
        rdkit.Chem.Mol: The constructed RDKit molecule.
    """
    mol = Chem.RWMol()

    for atom_features in graph["node_feat"]:
        atomic_number = int(
            atom_features[0]
        )
        mol.AddAtom(Chem.Atom(atomic_number))

    edge_index = graph["edge_index"]
    edge_feat = graph["edge_feat"]

    for i in range(edge_index.shape[1] // 2):
        src, tgt = edge_index[:, i]
        bond_type = Chem.rdchem.BondType.SINGLE

        if edge_feat[i][0] == 1:
            bond_type = Chem.rdchem.BondType.SINGLE
        elif edge_feat[i][1] == 1:
            bond_type = Chem.rdchem.BondType.DOUBLE
        elif edge_feat[i][2] == 1:
            bond_type = Chem.rdchem.BondType.TRIPLE
        elif edge_feat[i][0] == 3:
            bond_type = Chem.rdchem.BondType.AROMATIC

        if not mol.GetBondBetweenAtoms(int(src), int(tgt)):
            mol.AddBond(int(src), int(tgt), bond_type)

    return mol


def mol_visualize(mol):
    """
    Visualizes a molecule using RDKit.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object to visualize.

    Returns:
        None: Displays the molecule visualization.
    """
    from rdkit.Chem import Draw

    img = Draw.MolToImage(mol)
    img.show()


def make_specific_smarts(smarts):
    """
    Handles SMARTS patterns with multiple atom choices and operators,
    including exclusions like [!#6;!#1].

    Args:
        smarts (str): SMARTS pattern to process.

    Returns:
        str: A new SMARTS pattern with random atom choices selected.
    """
    pattern = r"(\[\$\(.*?\)\]|[~=#:@\-\+*]|1|\[[^]]+\])"
    parts = re.findall(
        pattern, smarts
    )

    result = []
    for part in parts:
        if part.startswith("[$(") and part.endswith(")]"):
            nested_options = part[2:-1].split(
                "),$("
            )
            selected_option = random.choice(
                nested_options
            )
            randomized_nested = make_specific_smarts(
                selected_option
            )
            result.append(randomized_nested)
        elif part.startswith("[") and part.endswith("]"):
            atom_group = part.strip("[]")
            if atom_group in atom_groups:
                selected_atom = random.choice(atom_groups[atom_group])
                result.append(f"[{selected_atom}]")
            elif (
                ";" in atom_group and "," in atom_group
            ):
                base, subgroups = atom_group.split(";")
                subgroup_choices = subgroups.split(",")
                selected_subgroup = random.choice(
                    subgroup_choices
                )
                result.append(f"[{base}{selected_subgroup}]")
            elif "," in atom_group:
                atom_choices = atom_group.split(",")
                selected_atom = random.choice(atom_choices)
                result.append(f"[{selected_atom}]")
            else:
                result.append(part)
        elif part == "*":
            selected_atom = random.choice(atom_groups["*"])
            result.append(f"[{selected_atom}]")
        elif part == "~" or part == "@":
            selected_operator = random.choice(atom_groups["~"])
            result.append(selected_operator)
        else:
            result.append(part)

    return "".join(result)


def add_replacement_mol(editable_mol, replacement_mol, attach_idx):
    """
    Adds all atoms and bonds from the replacement molecule to the editable molecule,
    and connects them with at least one bond.

    Args:
        editable_mol (rdkit.Chem.RWMol): The base molecule to modify.
        replacement_mol (rdkit.Chem.RWMol): The molecule to add.

    Returns:
        None
    """
    atom_map = {}

    for atom in replacement_mol.GetAtoms():
        new_idx = editable_mol.AddAtom(atom)
        atom_map[atom.GetIdx()] = new_idx

    for bond in replacement_mol.GetBonds():
        start_idx = atom_map[bond.GetBeginAtomIdx()]
        end_idx = atom_map[bond.GetEndAtomIdx()]
        editable_mol.AddBond(start_idx, end_idx, bond.GetBondType())

    editable_mol.AddBond(attach_idx, atom_map[0], Chem.BondType.SINGLE)

    editable_mol.UpdatePropertyCache(strict=False)
    return editable_mol


def extract_and_modify(selfies, replace_ratio=0.1, modify_num=0):
    smiles = sf.decoder(selfies)
    if not smiles:
        raise ValueError(f"Invalid SELFIES, SMILES provided: {selfies}, {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to convert smiles: {smiles} to a valid molecule.")

    maccs_keys = MACCSkeys.GenMACCSKeys(mol)

    active_keys_and_substructures = []
    for key, (smarts, _) in smartsPatts.items():
        if maccs_keys.GetBit(key):
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    substructures = [
                        Chem.MolFragmentToSmiles(mol, match) for match in matches
                    ]
                    atom_indices = [
                        [idx for idx in match] for match in matches
                    ]
                    active_keys_and_substructures.append(
                        {
                            "key": key,
                            "smarts": smarts,
                            "substructures": substructures,
                            "atom_indices": atom_indices,
                        }
                    )

    unused_smarts = []
    for key, (smarts, _) in smartsPatts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and not mol.HasSubstructMatch(pattern):
            unused_smarts.append(smarts)

    n = len(active_keys_and_substructures)
    if modify_num == 0:
        num_to_modify = max(int(replace_ratio * n), 1)
    else:
        num_to_modify = modify_num
    if num_to_modify == 0 or not active_keys_and_substructures or not unused_smarts:
        return {
            "original_smiles": smiles,
            "num_of_key_substructures": n,
            "non_removed_count": None,
            "non_added_count": None,
            "substructures_removed": None,
            "substructures_added": None,
            "removed_graph": None,
            "modified_smiles": "CCC",
            "modified_graph": mol2graph(Chem.MolFromSmiles("CCC")),
        }

    substructures_to_remove = random.sample(
        active_keys_and_substructures, num_to_modify
    )

    new_substructures = random.sample(unused_smarts, num_to_modify)

    editable_mol = Chem.RWMol(mol)

    not_removed_count = 0
    for substructure in substructures_to_remove:
        pattern = Chem.MolFromSmarts(substructure["smarts"])
        matches = editable_mol.GetSubstructMatches(pattern)
        if matches:
            for _ in range(len(matches)):
                matches = editable_mol.GetSubstructMatches(pattern)
                if matches:
                    try:
                        match = matches[0]
                        res = Chem.RWMol(editable_mol)
                        res.BeginBatchEdit()
                        for aid in match:
                            res.RemoveAtom(aid)
                        res.CommitBatchEdit()
                        Chem.SanitizeMol(res)
                        editable_mol = res
                    except:
                        continue
        else:
            not_removed_count += 1

    editable_mol = Chem.RWMol(editable_mol)
    removed_mol = mol2graph(editable_mol)

    non_added_count = 0
    for smarts in new_substructures:
        new_smarts = make_specific_smarts(smarts)
        replacement_mol = Chem.MolFromSmarts(new_smarts)
        if replacement_mol:
            replacement_mol = Chem.RWMol(replacement_mol)
            attach_idx = 0
            while attach_idx < editable_mol.GetNumAtoms() - 1:
                try:
                    copy_mol = editable_mol

                    new_atom_idx = copy_mol.AddAtom(replacement_mol.GetAtomWithIdx(0))
                    copy_mol.AddBond(attach_idx, new_atom_idx, Chem.BondType.SINGLE)
                    editable_mol.UpdatePropertyCache(strict=False)
                    editable_mol = copy_mol
                    break
                except:
                    attach_idx += 1
                    pass
            if attach_idx == editable_mol.GetNumAtoms() - 1:
                non_added_count += 1

    modified_smiles = Chem.MolToSmiles(editable_mol)
    modified_graph = mol2graph(editable_mol)
    if modified_graph["num_nodes"] == 0:
        return {
            "original_smiles": smiles,
            "num_of_key_substructures": n,
            "non_removed_count": not_removed_count,
            "non_added_count": non_added_count,
            "substructures_removed": substructures_to_remove,
            "substructures_added": new_substructures,
            "removed_graph": removed_mol,
            "modified_smiles": "CCC",
            "modified_graph": mol2graph(Chem.MolFromSmiles("CCC")),
        }
    assert modified_graph["num_nodes"] != 0, "Modified graph has no nodes."
    return {
        "original_smiles": smiles,
        "num_of_key_substructures": n,
        "non_removed_count": not_removed_count,
        "non_added_count": non_added_count,
        "substructures_removed": substructures_to_remove,
        "substructures_added": new_substructures,
        "removed_graph": removed_mol,
        "modified_smiles": modified_smiles,
        "modified_graph": modified_graph,
    }


def size_augmentation_single_mol(mol, min_r=0.3, max_r=0.9):
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


def get_unique_atoms_and_counts(molecule):
    """
    Get a list of unique atoms and their counts in the given RDKit molecule object.

    Args:
    molecule (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
    dict: A dictionary with atom symbols as keys and their counts as values.
    """
    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]

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
    atoms = list(atom_counts.keys())
    counts = list(atom_counts.values())

    sampled_atom = np.random.choice(atoms, p=np.array(counts) / np.sum(counts)).item()

    return sampled_atom


def add_atoms_based_on_mol(mol, num_atoms_to_add):
    assert num_atoms_to_add > 0, "num_atoms_to_add should be positive."

    sanitized_mol = Chem.RWMol(mol)
    atom_counts = get_unique_atoms_and_counts(sanitized_mol)

    while num_atoms_to_add > 0:
        potential_attaching_indices = []
        for atom in sanitized_mol.GetAtoms():
            all_single_bonds = all(
                [
                    bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                    for bond in atom.GetBonds()
                ]
            )

            if atom.GetImplicitValence() > 0 or not all_single_bonds:
                potential_attaching_indices.append(atom.GetIdx())

        if not potential_attaching_indices:
            break

        np.random.shuffle(potential_attaching_indices)

        change_made = False
        for atom_index in potential_attaching_indices:
            try:
                rw_mol = Chem.RWMol(sanitized_mol)
                new_atom = Chem.Atom(sample_atom(atom_counts))
                new_atom_idx = rw_mol.AddAtom(new_atom)
                attaching_atom = rw_mol.GetAtomWithIdx(atom_index)
                if (
                    attaching_atom.GetTotalValence()
                    <= attaching_atom.GetExplicitValence()
                ):
                    bonds = attaching_atom.GetBonds()
                    one_level_downgrade = {
                        Chem.rdchem.BondType.DOUBLE: Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.TRIPLE: Chem.rdchem.BondType.DOUBLE,
                    }
                    bonds = [
                        bond
                        for bond in bonds
                        if bond.GetBondType() in one_level_downgrade
                    ]
                    np.random.shuffle(bonds)
                    bond = bonds[0]
                    new_bond_type = one_level_downgrade[bond.GetBondType()]
                    rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    rw_mol.AddBond(
                        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_bond_type)
                    )

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


def map_by_substructure_replacement(
    data_point, replace_ratio=0.1, num_rejected_graphs=5
):

    replace_ratio = min(replace_ratio, 1.0)
    task = data_point["task"]
    input_mol_string = data_point["input_mol_string"]

    selfies = (
        input_mol_string.replace("<SELFIES>", "")
        .replace("</SELFIES>", "")
        .replace(" ", "")
    )

    for i in range(num_rejected_graphs):
        if task in REGRESSION_BENCHMARKS:
            prob = np.random.rand()
            if prob < 0.5:
                rejected_graph = extract_and_modify(
                    selfies, replace_ratio=replace_ratio
                )["modified_graph"]
                additional_rejected_graph = extract_and_modify(
                    selfies, replace_ratio=replace_ratio
                )["modified_graph"]
            else:
                smiles = sf.decoder(selfies)
                mol = Chem.MolFromSmiles(smiles)
                rejected_mol = size_augmentation_single_mol(mol)
                rejected_graph = mol2graph(rejected_mol["mol"])
                additional_rejected_graph = mol2graph(rejected_mol["mol"])

        elif "|>>|" in selfies:
            pair_selfies = selfies.split("|>>|")
            rejected_graph = extract_and_modify(
                pair_selfies[0], replace_ratio=replace_ratio
            )["modified_graph"]
            additional_rejected_graph = extract_and_modify(
                pair_selfies[1], replace_ratio=replace_ratio
            )["modified_graph"]
        elif task in TEXT2MOL_BENCHMARKS + [
            "smol-name_conversion-i2s",
            "smol-name_conversion-i2f",
        ]:
            dummy_selfies = "[C][C][C]"
            rejected_graph = extract_and_modify(
                dummy_selfies, replace_ratio=replace_ratio
            )["modified_graph"]
            additional_rejected_graph = extract_and_modify(
                dummy_selfies, replace_ratio=replace_ratio
            )["modified_graph"]
        else:
            rejected_graph = extract_and_modify(selfies, replace_ratio=replace_ratio)[
                "modified_graph"
            ]
            additional_rejected_graph = extract_and_modify(
                selfies, replace_ratio=replace_ratio
            )["modified_graph"]

        data_point[f"{i}-th_rejected_x"] = rejected_graph["node_feat"]
        data_point[f"{i}-th_rejected_edge_index"] = rejected_graph["edge_index"]
        data_point[f"{i}-th_rejected_edge_attr"] = rejected_graph["edge_feat"]
        data_point[f"{i}-th_additional_rejected_x"] = additional_rejected_graph[
            "node_feat"
        ]
        data_point[f"{i}-th_additional_rejected_edge_index"] = (
            additional_rejected_graph["edge_index"]
        )
        data_point[f"{i}-th_additional_rejected_edge_attr"] = additional_rejected_graph[
            "edge_feat"
        ]

    return data_point


if __name__ == "__main__":
    from data.builders.hard_negative.__main__ import main
    main()
