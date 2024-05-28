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


def smiles_handler(text, mol_ph, is_gal=True, graph_only=False):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    if graph_only:
        text = CUSTOM_SEQ_RE.sub(r"%s" % (mol_ph), text)
        return text, smiles_list
    if is_gal:
        text = CUSTOM_SEQ_RE.sub(r"\1\3\4%s" % (mol_ph), text)
        text = escape_custom_split_sequence(text)
        return text, smiles_list
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
        is_gal=True,
        graph_only=False,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        self.graph_only = graph_only

    def __call__(self, batch):
        graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)

        ## deal with prompt
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.is_gal, self.graph_only)[0]
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
        is_gal=True,
        graph_only=False,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        self.graph_only = graph_only

    def __call__(self, batch):
        graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.is_gal, self.graph_only)[0]
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


PROPERTY_CLASSIFICATION_BENCHMARKS = [
    "bace",
    "bbbp",
    "clintox",
    "toxcast",
    "sider",
    "tox21",  # molca
    "hiv",
    "pcba",
    "muv",
    "chembl",
]


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
        self.graph_only = args.graph_only

        if root in PROPERTY_CLASSIFICATION_BENCHMARKS:
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
        self.is_gal = args.opt_model.find("galactica") >= 0

    def get_dataset_from_deepchem(self, root):
        base_path = f"dataset/{root}"
        os.makedirs(base_path, exist_ok=True)
        # load tox21 dataset using deepchem

        if root == "bace":
            loading_fn = dc.molnet.load_bace_classification  # 1 task
        elif root == "bbbp":
            loading_fn = dc.molnet.load_bbbp  # 1 task
        elif root == "clintox":
            loading_fn = dc.molnet.load_clintox  # 2 tasks
        elif root == "toxcast":
            loading_fn = dc.molnet.load_toxcast  # 617
        elif root == "sider":
            loading_fn = dc.molnet.load_sider  # 27 tasks
        elif root == "tox21":
            loading_fn = dc.molnet.load_tox21  # 12 tasks
        elif root == "qm9":  # 12 tasks
            loading_fn = dc.molnet.load_qm9
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
                    self.is_gal,
                    self.graph_only,
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
                    self.is_gal,
                    self.graph_only,
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
                self.is_gal,
                self.graph_only,
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
                self.is_gal,
                self.graph_only,
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
                self.is_gal,
                self.graph_only,
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
        parser.add_argument("--graph_only", action="store_true", default=False)

        # moleculenet dataset
        parser.add_argument("--subtask_idx", type=int, default=0)
        return parent_parser


from tqdm import tqdm
from rdkit import Chem


class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(self, data, tasks, subtask_idx=0, prompt=None, debug=False):
        self.mol_list = data.X
        self.label_list = data.y[:, subtask_idx]
        self.tasks_list = tasks
        self.task = tasks[subtask_idx]
        if debug:
            self.mol_list = self.mol_list[:100]
            self.label_list = self.label_list[:100]
        self.prompt = prompt
        # label wrapping token
        self.bool_token = ["<BOOLEAN>", "</BOOLEAN>"]
        self.float_token = ["<FLOAT>", "</FLOAT>"]

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

    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        label = self.label_list[index]
        if label:
            label = self.bool_token[0] + "True" + self.bool_token[1]
        else:
            label = self.bool_token[0] + "False" + self.bool_token[1]
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
        instruction = f"\n Given a molecule from the {dataset_name} dataset, you are predicting whether the molecule has the {task_name} property. The answer should be in the form {self.bool_token[0]}True{self.bool_token[1]} or {self.bool_token[0]}False{self.bool_token[1]}."
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
