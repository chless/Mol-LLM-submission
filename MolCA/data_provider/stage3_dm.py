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
from torch_geometric.data import Data
import os

import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter

# we split individual characters inside special tokens like [START_DNA]
# TODO: change this ugly I_SMILES things to regular special token, and add the special token to vocab whichever LLM
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", rf"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph, mol_representation):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    # graph embedding without smiles tokens
    # '<mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    if mol_representation == "graph_only":
        text = CUSTOM_SEQ_RE.sub(r"%s" % (mol_ph), text)
        return text, smiles_list
    # smiles tokens without graph embedding
    # \1, \4 corresponds to the special tokens for string (\2 is the special token, which included in the nest of \1)
    # \3 corresponds to the content of the smiles token
    # '[START_I_SMILES][H]N([H])C(=O)C([H])([H])[H][END_I_SMILES]'
    elif mol_representation == "string_only":
        text = CUSTOM_SEQ_RE.sub(r"\1\3\4", text)
        return text, smiles_list
    # smiles tokens with graph embedding
    # '[START_I_SMILES][H]N([H])C(=O)C([H])([H])[H][END_I_SMILES]<mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    elif mol_representation == "string+graph":
        text = CUSTOM_SEQ_RE.sub(r"\1\3\4%s" % (mol_ph), text)
        text = escape_custom_split_sequence(text)
        return text, smiles_list
    # smiles tokens with graph tokens without special tokens
    # '[H]N([H])C(=O)C([H])([H])[H]<mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    else:
        text = CUSTOM_SEQ_RE.sub(r"\3%s" % (mol_ph), text)
        return text, smiles_list


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


class TrainCollater:
    def __init__(
        self,
        tokenizer,
        text_max_len,
        mol_ph,
        mol_token_id,
        mol_representation=True,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation

    def __call__(self, batch):
        graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)

        ## deal with prompt
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.mol_representation)[0]
            for p in smiles_prompt
        ]

        self.tokenizer.padding_side = "left"
        smiles_prompt_tokens = self.tokenizer(
            text=smiles_prompt,
            truncation=False,
            padding="longest",
            add_special_tokens=True,
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
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation

    def __call__(self, batch):
        graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.mol_representation)[0]
            for p in smiles_prompt
        ]

        ## deal with prompt
        self.tokenizer.padding_side = "left"
        smiles_prompt_tokens = self.tokenizer(
            smiles_prompt,
            return_tensors="pt",
            add_special_tokens=True,
            # max_length=self.text_max_len,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
        )

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens["is_mol_token"] = is_mol_token
        return graphs, smiles_prompt_tokens, texts, tasks


# binary classification
PROPERTY_CLASSIFICATION_BENCHMARKS = [
    "bace",  # 1 task # molca, biot5+, instructmol
    "bbbp",  # 1 task # molca, biot5+, instructmol, llasmol
    "clintox",  # 2 tasks # molca, biot5+, llasmol
    "toxcast",  # 617 # molca
    "sider",  # 27 # molca, llasmol
    "tox21",  # 12 tasks # molca
    "hiv",  # 1 tasks # biot5+, instructmol, llasmol
]
PROPERTY_REGRESSION_BENCHMARKS = [
    "qm9",  # 12 tasks #biot5+, instructmol (homo:2, lumo:3, homo-lumo gap:4)
    "esol",  # 1 task # llasmol
    "lipo",  # 1 task # llasmol
]

CAPTIONING_BENCHMARKS = ["pubchem324k"]

# INSTRUCTION_TEMPLATE = "\n Given a molecule from the {dataset_name} dataset, you are predicting whether the molecule has the {task_name} property. The answer should be in the form {label_tokens[0]}True{label_tokens[1]} or {label_tokens[0]}False{label_tokens[1]}."
INSTRUCTION_CLASSIFICATION = "\n Given the molecule, you should predict {task_name} property of the molecule. The answer should be in the form {label_tokens[0]}True{label_tokens[1]} or {label_tokens[0]}False{label_tokens[1]}."
INSTRUCTION_REGRESSION = "\n Given the molecule, you should predict {task_name} property of the molecule. The answer should be in the form {label_tokens[0]}x.xxxx{label_tokens[1]}."
INSTRUCTION_CAPTIONING = "\n Given the molecule, you should generate a text description corresponding to the molecule. The answer should be in the form {label_tokens[0]}text{label_tokens[1]}."
INSTRUCTION_REACTION = "\n Given the molecule, you should predict chemical reaction involving the molecule. The answer should be in the form {label_tokens[0]}SMILES{label_tokens[1]}."


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

        if root in PROPERTY_CLASSIFICATION_BENCHMARKS + PROPERTY_REGRESSION_BENCHMARKS:
            self.tasks, self.train_data, self.val_data, self.test_data = (
                self.get_dataset_from_deepchem(root)
            )
            self.tasks = [f"{root}/{t}" for t in self.tasks]
            self.train_dataset = MoleculeNetDatasetDeepChem(
                data=self.train_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=args.debug,
            )
            self.val_dataset = MoleculeNetDatasetDeepChem(
                data=self.val_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=args.debug,
            )
            self.test_dataset = MoleculeNetDatasetDeepChem(
                data=self.test_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=args.debug,
            )
        elif root in ["molnet_cls"]:
            datasets = [
                self.get_dataset_from_deepchem(task_name)
                for task_name in PROPERTY_CLASSIFICATION_BENCHMARKS
            ]
            self.tasks, self.train_data, self.val_data, self.test_data = zip(*datasets)
            # TODO:replace hard coding for cases, which have multiple tasks

            self.tasks = [f"{root}/{t}" for t in self.tasks]
            self.train_dataset = MoleculeNetDatasetDeepChem(
                data=self.train_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=args.debug,
            )
            self.val_dataset = MoleculeNetDatasetDeepChem(
                data=self.val_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=args.debug,
            )
            self.test_dataset = MoleculeNetDatasetDeepChem(
                data=self.test_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=args.debug,
            )
        else:
            self.pretrain_dataset = MoleculeCaptionV2(
                root + f"pretrain.pt", text_max_len, self.prompt, debug=args.debug
            )
            self.train_dataset = MoleculeCaptionV2(
                root + f"train.pt", text_max_len, self.prompt, debug=args.debug
            )
            self.val_dataset = MoleculeCaptionV2(
                root + f"valid.pt", text_max_len, self.prompt, debug=args.debug
            )
            self.test_dataset = MoleculeCaptionV2(
                root + f"test.pt", text_max_len, self.prompt, debug=args.debug
            )

        self.init_tokenizer(tokenizer)
        self.mol_ph_token = "<mol>" * self.args.num_query_token
        self.mol_representation = args.mol_representation

    def get_dataset_from_deepchem(self, root):
        base_path = f"dataset/{root}"
        os.makedirs(base_path, exist_ok=True)
        # load tox21 dataset using deepchem

        if root == "bace":
            loading_fn = dc.molnet.load_bace_classification
        elif root == "esol":
            loading_fn = dc.molnet.load_delaney
        elif (
            root
            in PROPERTY_CLASSIFICATION_BENCHMARKS
            + PROPERTY_REGRESSION_BENCHMARKS
            + CAPTIONING_BENCHMARKS
        ):
            loading_fn = getattr(dc.molnet, f"load_{root}")
        else:
            raise NotImplementedError

        tasks, datasets, transformers = loading_fn(
            featurizer="Raw",
            splitter="scaffold",
            save_dir=base_path,
            data_dir=base_path,
            reload=True,
        )
        train_dataset, valid_dataset, test_dataset = datasets
        return tasks, train_dataset, valid_dataset, test_dataset

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == "pretrain":
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                ),
            )
        elif self.mode == "ft":
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                ),
            )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(
                self.tokenizer,
                self.text_max_len,
                self.mol_ph_token,
                self.mol_token_id,
                self.mol_representation,
            ),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(
                self.tokenizer,
                self.text_max_len,
                self.mol_ph_token,
                self.mol_token_id,
                self.mol_representation,
            ),
        )
        return [val_loader, test_loader]

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(
                self.tokenizer,
                self.text_max_len,
                self.mol_ph_token,
                self.mol_token_id,
                self.mol_representation,
            ),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--inference_batch_size", type=int, default=4)
        parser.add_argument("--use_smiles", action="store_true", default=False)
        parser.add_argument("--root", type=str, default="data/PubChemDataset_v4")
        parser.add_argument("--text_max_len", type=int, default=128)
        parser.add_argument(
            "--prompt",
            type=str,
            default="The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ",
        )
        parser.add_argument("--filtered_cid_path", type=str, default=None)

        # moleculenet dataset
        parser.add_argument("--subtask_idx", type=int, default=0)
        return parent_parser


from tqdm import tqdm
from rdkit import Chem

BOOL_TOKENS = ["<BOOLEAN>", "</BOOLEAN>"]
FLOAT_TOKENS = ["<FLOAT>", "</FLOAT>"]


class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(self, data, tasks, subtask_idx=0, prompt=None, debug=False):
        self.mol_list = data.X
        self.label_list = data.y[:, subtask_idx]
        self.tasks_list = tasks
        self.root = tasks[0].split("/")[0]
        self.task = tasks[subtask_idx]
        if debug:
            self.mol_list = self.mol_list[:100]
            self.label_list = self.label_list[:100]
        self.prompt = prompt
        # label wrapping token
        if self.root in PROPERTY_CLASSIFICATION_BENCHMARKS:
            self.label_tokens = BOOL_TOKENS
        elif self.root in PROPERTY_REGRESSION_BENCHMARKS:
            self.label_tokens = FLOAT_TOKENS
        else:
            raise NotImplementedError

        if not prompt:
            self.prompt = (
                "The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. "
            )
        else:
            self.prompt = prompt

        self.smiles_list = []
        iter_bar = tqdm(self.mol_list)
        for mol in iter_bar:
            self.smiles_list.append(Chem.MolToSmiles(mol))

    def __len__(self):
        return len(self.smiles_list)

    def wrap_label(self, label):
        if self.root in PROPERTY_CLASSIFICATION_BENCHMARKS:
            if label:
                return self.label_tokens[0] + "True" + self.label_tokens[1]
            else:
                return self.label_tokens[0] + "False" + self.label_tokens[1]
        elif self.root in PROPERTY_REGRESSION_BENCHMARKS:
            return self.label_tokens[0] + str(label) + self.label_tokens[1]
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        label = self.label_list[index]
        label = self.wrap_label(label)
        graph = smiles2data(smiles)

        if self.prompt.find("{}") >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        task = self.task
        smiles_prompt += self.get_instruction_for_task(task)

        return graph, label, smiles_prompt, task

    def get_instruction_for_task(self, task):
        dataset_name, task_name = task.split("/")
        if dataset_name in PROPERTY_CLASSIFICATION_BENCHMARKS:
            instruction = INSTRUCTION_CLASSIFICATION
        elif dataset_name in PROPERTY_REGRESSION_BENCHMARKS:
            instruction = INSTRUCTION_REGRESSION
        elif dataset_name in CAPTIONING_BENCHMARKS:
            instruction = INSTRUCTION_CAPTIONING
        else:
            raise NotImplementedError

        instruction = instruction.format(
            task_name=task_name,
            label_tokens=self.label_tokens,
        )
        return instruction

    def convert_selfies2smiles(self, selfies):
        from selfies import decoder

        return decoder(selfies)


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
