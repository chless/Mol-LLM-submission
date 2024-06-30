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
from torch_geometric.data import InMemoryDataset, Data
import os

import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
from torch.utils.data import ConcatDataset
from datasets import load_dataset

# we split individual characters inside special tokens like [START_DNA]
# TODO: change this ugly I_SMILES things to regular special token, and add the special token to vocab whichever LLM
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

BOOL_TOKENS = ["<BOOLEAN>", "</BOOLEAN>"]
FLOAT_TOKENS = ["<FLOAT>", "</FLOAT>"]
DESCRIPTION_TOKENS = ["<DESCRIPTION>", "</DESCRIPTION>"]
SMILES_TOKENS = ["[START_I_SMILES]", "[END_I_SMILES]"]


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


def smiles_handler(text, mol_ph, mol_representation, model="llama"):
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
        if "galactica" in model:
            text = escape_custom_split_sequence(text)
        return text, smiles_list
    # smiles tokens with graph embedding
    # '[START_I_SMILES][H]N([H])C(=O)C([H])([H])[H][END_I_SMILES]<mol><mol><mol><mol><mol><mol><mol><mol>.' + TEXT
    elif mol_representation == "string+graph":
        text = CUSTOM_SEQ_RE.sub(r"\1\3\4%s" % (mol_ph), text)
        if "galactica" in model:
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
        multi_task=False,
        model=None,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation
        self.multi_task = multi_task
        self.model = model

    def __call__(self, batch):
        # in multi-task, perdevice  batch size should be multiple of 4: classificaiton, regression, translation, reaction
        if self.multi_task:
            graphs, texts, smiles_prompt, tasks, instructions = zip(*batch)
        else:
            graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)

        ## deal with prompt
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.mol_representation, self.model)[0]
            for p in smiles_prompt
        ]

        for i in range(len(texts)):
            smiles_prompt[i] += instructions[i]

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
        multi_task=False,
        model=None,
    ):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.mol_representation = mol_representation
        self.multi_task = multi_task
        self.model = model

    def __call__(self, batch):
        if self.multi_task:
            graphs, texts, smiles_prompt, tasks, instructions = zip(*batch)
        else:
            graphs, texts, smiles_prompt, tasks = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [
            smiles_handler(p, self.mol_ph, self.mol_representation, self.model)[0]
            for p in smiles_prompt
        ]

        for i in range(len(texts)):
            smiles_prompt[i] += instructions[i]

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
        texts = self.tokenizer(
            text=texts,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.text_max_len,
            truncation=True,
            padding="longest",
            return_attention_mask=True,
        )

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens["is_mol_token"] = is_mol_token
        return graphs, smiles_prompt_tokens, texts, tasks


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
    "qm9",  # 12 tasks #biot5+, instructmol (homo:2, lumo:3, homo-lumo gap:4)
    "esol",  # 1 task # llasmol
    "lipo",  # 1 task # llasmol
]

MOL2TEXT_BENCHMARKS = ["molecular_description_generation"]

TEXT2MOL_BENCHMARKS = [
    "description_guided_molecule_design",
]

REACTION_BENCHAMRKS = [
    "forward_reaction_prediction",
    # "reagent_prediction", # TODO: deal with two molecule in input
    "retrosynthesis",
]

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
        self.debug = args.debug
        self.args = args

        if root == "multi_task":
            # preprocess dataset and save
            if self.args.get_raw_data:
                self.get_raw_multi_task_dataset()

            self.concat_datasets = {
                task: {"train": None, "val": None, "test": None}
                for task in ["classification", "regression", "reaction", "translation"]
            }

            for task in ["classification", "regression", "reaction", "translation"]:
                for split in ["train", "val", "test"]:
                    self.concat_datasets[task][split] = InstructionInMemoryDataset(
                        root=self.args.raw_data_root,
                        filename=f"{task}_{split}",
                    )

        elif root in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            self.tasks, self.train_data, self.val_data, self.test_data = (
                self.get_dataset(root)
            )
            self.tasks = [f"{root}/{t}" for t in self.tasks]
            self.train_dataset = MoleculeNetDatasetDeepChem(
                data=self.train_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=self.debug,
            )
            self.val_dataset = MoleculeNetDatasetDeepChem(
                data=self.val_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=self.debug,
            )
            self.test_dataset = MoleculeNetDatasetDeepChem(
                data=self.test_data,
                tasks=self.tasks,
                prompt=self.prompt,
                subtask_idx=args.subtask_idx,
                debug=self.debug,
            )
        else:
            self.pretrain_dataset = MoleculeCaptionV2(
                root + f"pretrain.pt", text_max_len, self.prompt, debug=self.debug
            )
            self.train_dataset = MoleculeCaptionV2(
                root + f"train.pt", text_max_len, self.prompt, debug=self.debug
            )
            self.val_dataset = MoleculeCaptionV2(
                root + f"valid.pt", text_max_len, self.prompt, debug=self.debug
            )
            self.test_dataset = MoleculeCaptionV2(
                root + f"test.pt", text_max_len, self.prompt, debug=self.debug
            )

        self.init_tokenizer(tokenizer)
        self.mol_ph_token = "<mol>" * self.args.num_query_token
        self.mol_representation = args.mol_representation

    def get_raw_multi_task_dataset(
        self,
    ):
        task_subtask_lists = {
            "bace": [0],
            "bbbp": [0],
            "clintox": [0, 1],
            "toxcast": [0],
            "sider": [0],
            "tox21": [0],
            "hiv": [0],
            "qm9": [2, 3, 4],
            "esol": [0],
            "lipo": [0],
            "forward_reaction_prediction": [0],
            # "reagent_prediction": [0],
            "retrosynthesis": [0],
            "description_guided_molecule_design": [0],
            "molecular_description_generation": [0],
        }
        self.task_subtask_pairs = [
            (task, subtask)
            for task, subtasks in task_subtask_lists.items()
            for subtask in subtasks
        ]

        total_benchmarks = (
            REACTION_BENCHAMRKS
            + MOL2TEXT_BENCHMARKS
            + TEXT2MOL_BENCHMARKS
            + CLASSIFICATION_BENCHMARKS
            + REGRESSION_BENCHMARKS
        )

        multi_task_datasets = {
            task_name: self.get_dataset(task_name)  # {task_name: [train, val, test]
            for task_name in total_benchmarks
        }

        self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
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

            if task_name in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
                train_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
                valid_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
                test_dataset = MoleculeNetDatasetDeepChem(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    subtask_idx=subtask_idx,
                    debug=self.debug,
                )
            elif task_name in REACTION_BENCHAMRKS:
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
            elif task_name in MOL2TEXT_BENCHMARKS:
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
            elif task_name in TEXT2MOL_BENCHMARKS:
                train_dataset = MolInstructionDatset(
                    data=data_split[0],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
                valid_dataset = MolInstructionDatset(
                    data=data_split[1],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )
                test_dataset = MolInstructionDatset(
                    data=data_split[2],
                    task_subtask_pair=task_subtask_pair,
                    prompt=self.prompt,
                    representation="smiles",
                    debug=self.debug,
                )

            self.train_dataset.append(train_dataset)
            self.val_dataset.append(valid_dataset)
            self.test_dataset.append(test_dataset)

        # concat datasets from each subtask into large class [classification, regression, reaction prediction, translation]

        concat_datasets = {
            task: {"train": [], "val": [], "test": []}
            for task in ["classification", "regression", "reaction", "translation"]
        }

        for i in range(len(self.task_subtask_pairs)):
            task_subtask_pair = self.task_subtask_pairs[i]
            task_name = task_subtask_pair[0]
            if task_name in CLASSIFICATION_BENCHMARKS:
                concat_datasets["classification"]["train"].append(self.train_dataset[i])
                concat_datasets["classification"]["val"].append(self.val_dataset[i])
                concat_datasets["classification"]["test"].append(self.test_dataset[i])
            elif task_name in REGRESSION_BENCHMARKS:
                concat_datasets["regression"]["train"].append(self.train_dataset[i])
                concat_datasets["regression"]["val"].append(self.val_dataset[i])
                concat_datasets["regression"]["test"].append(self.test_dataset[i])
            elif task_name in MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS:
                concat_datasets["translation"]["train"].append(self.train_dataset[i])
                concat_datasets["translation"]["val"].append(self.val_dataset[i])
                concat_datasets["translation"]["test"].append(self.test_dataset[i])
            elif task_name in REACTION_BENCHAMRKS:
                concat_datasets["reaction"]["train"].append(self.train_dataset[i])
                concat_datasets["reaction"]["val"].append(self.val_dataset[i])
                concat_datasets["reaction"]["test"].append(self.test_dataset[i])
            else:
                raise NotImplementedError

        os.path.makedirs(os.path.join(self.args.raw_data_root, "raw"), exist_ok=True)

        for task in ["classification", "regression", "reaction", "translation"]:
            for split in ["train", "val", "test"]:
                concat_dataset = ConcatDataset(concat_datasets[task][split])
                torch.save(
                    concat_dataset,
                    f"{self.args.raw_data_root}/raw/{task}_{split}",
                )

        print("Data preprocessing is done")
        return concat_datasets

    def get_dataset(self, root):
        base_path = f"dataset/{root}"
        os.makedirs(base_path, exist_ok=True)
        # get dataset from deepchem
        if root == "bace":
            loading_fn = dc.molnet.load_bace_classification
        elif root == "esol":
            loading_fn = dc.molnet.load_delaney
        elif root in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            loading_fn = getattr(dc.molnet, f"load_{root}")
        elif root in MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS:
            mol_instruction_dataset = load_dataset(
                "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
            )
            dataset = mol_instruction_dataset[root]
            train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
            split = train_dataset.train_test_split(test_size=0.1, shuffle=True)
            train_dataset, valid_dataset = split["train"], split["test"]

            test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
            tasks = [root]
        elif root in REACTION_BENCHAMRKS:
            mol_instruction_dataset = load_dataset(
                "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
            )
            dataset = mol_instruction_dataset[root]
            train_dataset = dataset.filter(lambda x: "train" in x["metadata"])
            split = train_dataset.train_test_split(test_size=0.1, shuffle=True)
            train_dataset, valid_dataset = split["train"], split["test"]

            test_dataset = dataset.filter(lambda x: "test" in x["metadata"])
            tasks = [root]
        else:
            raise NotImplementedError

        if root in CLASSIFICATION_BENCHMARKS + REGRESSION_BENCHMARKS:
            tasks, datasets, transformers = loading_fn(
                featurizer="Raw",
                splitter="scaffold",
                save_dir=base_path,
                data_dir=base_path,
                reload=True,
            )
            train_dataset, valid_dataset, test_dataset = datasets
        else:
            pass
        return tasks, train_dataset, valid_dataset, test_dataset

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # self.train_dataset.tokenizer = tokenizer
        # self.val_dataset.tokenizer = tokenizer
        # self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == "pretrain":
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
        elif self.mode == "ft":
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
        elif self.mode == "multi_task":
            loader = [
                DataLoader(
                    self.concat_datasets[task]["train"],
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=True,
                    collate_fn=TrainCollater(
                        self.tokenizer,
                        self.text_max_len,
                        self.mol_ph_token,
                        self.mol_token_id,
                        self.mol_representation,
                        multi_task=True,
                        model=self.args.opt_model,
                    ),
                )
                for task in ["classification", "regression", "reaction", "translation"]
            ]
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        if self.mode != "multi_task":
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                collate_fn=TrainCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.inference_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                collate_fn=InferenceCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
            return [val_loader, test_loader]
        else:
            loader = [
                DataLoader(
                    self.concat_datasets[task]["val"],
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=InferenceCollater(
                        self.tokenizer,
                        self.text_max_len,
                        self.mol_ph_token,
                        self.mol_token_id,
                        self.mol_representation,
                        multi_task=True,
                        model=self.args.opt_model,
                    ),
                )
                for task in ["classification", "regression", "reaction", "translation"]
            ]
            return loader

    def test_dataloader(self):
        if self.mode != "multi_task":
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.inference_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                collate_fn=InferenceCollater(
                    self.tokenizer,
                    self.text_max_len,
                    self.mol_ph_token,
                    self.mol_token_id,
                    self.mol_representation,
                    model=self.args.opt_model,
                ),
            )
            return loader
        else:
            loader = [
                DataLoader(
                    self.concat_datasets[task]["test"],
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=InferenceCollater(
                        self.tokenizer,
                        self.text_max_len,
                        self.mol_ph_token,
                        self.mol_token_id,
                        self.mol_representation,
                        multi_task=True,
                        model=self.args.opt_model,
                    ),
                )
                for task in ["classification", "regression", "reaction", "translation"]
            ]
            return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=4)
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
        parser.add_argument("--get_raw_data", action="store_true", default=False)
        parser.add_argument(
            "--raw_data_root", type=str, default="MolCA/data/multi_task_dataset"
        )
        return parent_parser


from tqdm import tqdm

from rdkit import Chem


def wrap_label(label, task):

    if task in CLASSIFICATION_BENCHMARKS:
        label_tokens = BOOL_TOKENS
    elif task in REGRESSION_BENCHMARKS:
        label_tokens = FLOAT_TOKENS
    elif task in MOL2TEXT_BENCHMARKS:
        label_tokens = DESCRIPTION_TOKENS
    elif task in TEXT2MOL_BENCHMARKS + REACTION_BENCHAMRKS:
        label_tokens = SMILES_TOKENS
    else:
        raise NotImplementedError

    if task in CLASSIFICATION_BENCHMARKS:
        if label:
            return label_tokens[0] + "True" + label_tokens[1]
        else:
            return label_tokens[0] + "False" + label_tokens[1]
    elif task in REGRESSION_BENCHMARKS:
        return label_tokens[0] + str(label) + label_tokens[1]
    elif task in REACTION_BENCHAMRKS + MOL2TEXT_BENCHMARKS + TEXT2MOL_BENCHMARKS:
        return label_tokens[0] + label + label_tokens[1]
    else:
        raise NotImplementedError


from torch_geometric.data import InMemoryDataset, Data


# TODO use task or refactor it
# getitem shoul return enough information that what is label, and what is the label meaning (to format instruction)
class MoleculeNetDatasetDeepChem(Dataset):
    def __init__(
        self, data, task_subtask_pair, subtask_idx=0, prompt=None, debug=False
    ):
        self.debug = debug
        self.data = data
        self.subtask_idx = subtask_idx
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        if self.task in CLASSIFICATION_BENCHMARKS:
            instruction = INSTRUCTION_CLASSIFICATION
            self.label_tokens = BOOL_TOKENS
        elif self.task in REGRESSION_BENCHMARKS:
            instruction = INSTRUCTION_REGRESSION
            self.label_tokens = FLOAT_TOKENS
        else:
            raise NotImplementedError

        self.instruction = instruction.format(
            task_name=self.subtask,
            label_tokens=self.label_tokens,
        )

        if not prompt:
            self.prompt = (
                "The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. "
            )
        else:
            self.prompt = prompt

        self.set_necessary_data()

    def __len__(self):
        return len(self.smiles_list)

    def get_necessary_data(self, index):
        smiles = self.smiles_list[index]
        label = self.label_list[index]
        label = wrap_label(label, self.task)
        graph = smiles2data(smiles)

        if self.prompt.find("{}") >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt

        return graph, label, smiles_prompt, self.instruction

    def set_necessary_data(self):
        self.mol_list = self.data.X
        self.label_list = self.data.y[:, self.subtask_idx]
        if self.debug:
            self.mol_list = self.mol_list[:100]
            self.label_list = self.label_list[:100]

        self.smiles_list = []
        for mol in self.mol_list:
            self.smiles_list.append(Chem.MolToSmiles(mol))

        label_list = []
        smiles_prompt_list = []
        graph_list = []

        self.count_invalid_smiles = 0

        for i in range(len(self.mol_list)):
            try:
                graph, label, smiles_prompt, instruction = self.get_necessary_data(i)
                label_list.append(label)
                smiles_prompt_list.append(smiles_prompt)
                graph_list.append(graph)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles/len(self.input_list)}"
            )

        self.label_list = label_list
        self.smiles_prompt_list = smiles_prompt_list
        self.graph_list = graph_list

    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        smiles_prompt = self.smiles_prompt_list[index]

        return graph, label, smiles_prompt, self.task_subtask_pair, self.instruction


from selfies import decoder


def convert_selfies2smiles(selfies):
    return decoder(selfies)


from tqdm import tqdm


class MolInstructionDatset(Dataset):
    def __init__(
        self, data, task_subtask_pair, prompt=None, representation="smiles", debug=False
    ):
        self.debug = debug
        self.data = data
        self.prompt = prompt
        self.representation = representation
        self.task_subtask_pair = task_subtask_pair
        self.task, self.subtask = task_subtask_pair.split("/")

        if not prompt:
            self.prompt = (
                "The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. "
            )
        else:
            self.prompt = prompt

        self.set_necesary_data()

    def set_necesary_data(self):
        if self.debug:
            self.data = self.data[:100]
        else:
            self.data = self.data

        self.input_list = self.data["input"]
        self.label_list = self.data["output"]
        self.instruction_list = self.data["instruction"]

        input_list = []
        label_list = []
        smiles_prompt_list = []
        graph_list = []
        instruction_list = []

        self.count_invalid_smiles = 0
        for i in range(len(self.input_list)):
            try:
                graph, label, smiles_prompt, instruction = self.get_necessary_data(i)
                input_list.append(self.input_list[i])
                label_list.append(label)
                smiles_prompt_list.append(smiles_prompt)
                graph_list.append(graph)
                instruction_list.append(instruction)
            except Exception as e:
                self.count_invalid_smiles += 1
        if self.count_invalid_smiles > 0:
            print(f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles}")
            print(
                f"{self.task}: Number of invalid smiles: {self.count_invalid_smiles/len(self.input_list)}"
            )

        self.input_list = input_list
        self.label_list = label_list
        self.smiles_prompt_list = smiles_prompt_list
        self.graph_list = graph_list
        self.instruction_list = instruction_list

    def __len__(self):
        return len(self.input_list)

    def get_necessary_data(self, index):
        instruction = self.instruction_list[index]
        input = self.input_list[index]
        label = self.label_list[index]
        # one smiles in output
        if self.task in TEXT2MOL_BENCHMARKS:
            selfies = label  # label in mol-instruction dataset is annotated as selfies
            smiles = convert_selfies2smiles(
                label
            )  # molca task smiles instead of selfies
            label = smiles
            # output smiles do not need to be converted to graph
            # but assign graph = None retrieve error in torch_geometric, so assign arbitral graph
            graph = smiles2data(
                "C"
            )  # output smiles do not need to be converted to graph
        # two smiles in input
        elif self.task in ["reagent_prediction"]:
            two_selfies = input
            list_selfies = two_selfies.split(">>")
            smiles = [convert_selfies2smiles(s) for s in list_selfies]
            graph = [smiles2data(s) for s in smiles]
        else:
            # one smiles in input
            selfies = input
            smiles = convert_selfies2smiles(selfies)
            graph = smiles2data(smiles)

        label = wrap_label(label, self.task)

        if self.prompt.find("{}") >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt

        return graph, label, smiles_prompt, instruction

    # LLM input order: <instruction><qformer_output><smiles_tokens>
    def __getitem__(self, index):
        graph = self.graph_list[index]
        label = self.label_list[index]
        smiles_prompt = self.smiles_prompt_list[index]
        instruction = self.instruction_list[index]

        return graph, label, smiles_prompt, self.task_subtask_pair, instruction


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


# Initialize with the data_list from ConcatDataset
class InstructionInMemoryDataset(InMemoryDataset):
    def __init__(self, root, filename, transform=None, pre_transform=None, debug=False):
        self.filename = filename  # raw_file_names and processed_file_names use this
        super(InstructionInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        if debug:
            self.reduce_dataset_size(100)

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

        self.data = Data(**reduced_data)

    @property
    def raw_file_names(self):
        return f"{self.filename}.pth"

    @property
    def processed_file_names(self):
        return f"{self.filename}.pt"

    def download(self):
        raise NotImplementedError(
            "Data is not available for download. Preprocess data and save it."
        )

    def process(self):
        # Process data_list and store in `self.data` and `self.slices`
        raw_data_list = list(torch.load(self.raw_paths[0]))
        data_list = []

        for instance in raw_data_list:
            data = Data(
                x=instance[0].x,
                edge_index=instance[0].edge_index,
                edge_attr=instance[0].edge_attr,
                y=instance[1],
                smiles_prompt=instance[2],
                task_subtask_pair=instance[3],
                instruction=instance[4],
            )

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __getitem__(self, index):
        data = self.get(index)
        label = data.y
        smiles_prompt = data.smiles_prompt
        task_subtask_pair = data.task_subtask_pair
        instruction = data.instruction
        return data, label, smiles_prompt, task_subtask_pair, instruction
