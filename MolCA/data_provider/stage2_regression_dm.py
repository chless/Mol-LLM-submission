# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
import re
from ogb.utils import smiles2graph
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


INSTRUCTIONS = {
    "HOMO": [
        'I would like to know the highest occupied molecular orbital (HOMO) energy of this molecule, could you please provide it?',
        'Please provide the HOMO energy value for this molecule.',
        'I am interested in the HOMO energy of this molecule, could you tell me what it is?',
        'What is the highest occupied molecular orbital (HOMO) energy of this molecule?',
        'Could you give me the HOMO energy value of this molecule?',
        'What is the HOMO energy of this molecule?',
        'Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.',
        'Please provide me with the HOMO energy value of this molecule.',
        'What is the HOMO level of energy for this molecule?',
        'I would like to know the HOMO energy of this molecule, could you please provide it?',
        'Can you tell me the value of the HOMO energy for this molecule?',
        'Please provide the highest occupied molecular orbital (HOMO) energy of this molecule.',
    ],
    "LUMO": [
        'Please provide me with the LUMO energy value of this molecule.',
        'I am interested in the LUMO energy of this molecule, could you tell me what it is?',
        'I would like to know the lowest unoccupied molecular orbital (LUMO) energy of this molecule, could you please provide it?',
        'What is the LUMO energy of this molecule?',
        'What is the LUMO level of energy for this molecule?',
        'I would like to know the LUMO energy of this molecule, could you please provide it?',
        'What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?',
        'Could you give me the LUMO energy value of this molecule?',
        'Please provide the lowest unoccupied molecular orbital (LUMO) energy value for this molecule.',
        'Please provide the lowest unoccupied molecular orbital (LUMO) energy of this molecule.',
        'Can you tell me the value of the LUMO energy for this molecule?',
        'Please provide the LUMO energy value for this molecule.',
    ],
    "HOMO-LUMO-gap": [
        'Please provide the gap between HOMO and LUMO of this molecule.',
        'I would like to know the HOMO-LUMO gap of this molecule, can you provide it?',
        'Please give me the HOMO-LUMO gap energy for this molecule.',
        'Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule?',
        'Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.',
        'I need to know the HOMO-LUMO gap energy of this molecule, could you please provide it?',
        'What is the energy separation between the HOMO and LUMO of this molecule?',
        'Could you tell me the energy difference between HOMO and LUMO for this molecule?',
        'What is the HOMO-LUMO gap of this molecule?',
    ]
}


def smiles_handler(text, mol_ph, is_gal=True):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    if is_gal:
        text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
        text = escape_custom_split_sequence(text)
        return text, smiles_list
    else:
        text = CUSTOM_SEQ_RE.sub(r'\3%s' % (mol_ph), text)
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
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, is_gal=True):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        
        ## deal with prompt
        smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in smiles_prompt]
        # prompt_tokens = self.tokenizer(smiles_prompt, return_tensors='pt', max_length=self.text_max_len, padding='longest', truncation=True, return_attention_mask=True)
        # prompt_lens = prompt_tokens.attention_mask.sum(dim=1)

        # smiles_prompt = [p) for p in smiles_prompt]
        ## concate text and prompt

        # texts = [escape_custom_split_sequence(prompt + text) for prompt, text in zip(smiles_prompt, texts)]
        self.tokenizer.paddding_side = 'left'
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        # print(smiles_prompt_tokens.input_ids, self.mol_token_id)
        # print(is_mol_token)
        self.tokenizer.paddding_side = 'right'
        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return graphs, smiles_prompt_tokens, text_tokens

    

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, is_gal=True):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in smiles_prompt]
        ## deal with prompt
        self.tokenizer.paddding_side = 'left'
        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                       return_tensors='pt', 
                                    #    max_length=self.text_max_len, 
                                       padding='longest', 
                                       truncation=False, 
                                       return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        return graphs, smiles_prompt_tokens, texts
    

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

class RegressionDataset(Dataset):
    def __init__(self, data, prompt=None, representation='selfies'):
        self.data = data
        self.prompt = prompt
        self.representation = representation

        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt
        
        self.smiles_list = []
        self.label_list = []
        from tqdm import tqdm
        iter_bar = tqdm(self.data)
        for data in iter_bar:
            molecule, label = data['input'], data['output']
            self.smiles_list.append(molecule)
            self.label_list.append(label)

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, index):
        if self.representation == 'selfies':
            smiles = self.convert_selfies2smiles(self.smiles_list[index])
        else:
            smiles = self.smiles_list[index]
        label = self.label_list[index]
        graph = smiles2data(smiles)

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return graph, label, smiles_prompt
    
    def convert_selfies2smiles(self, selfies):
        from selfies import decoder
        return decoder(selfies)


class Stage2RegressionDM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
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

        data = self.get_external_data(root)
        train_dataset = data.filter(lambda x: 'train' in x['metadata'])
        split = train_dataset.train_test_split(test_size=0.1, shuffle=True)
        self.train_data, self.val_data = split['train'], split['test']

        # debug
        split = train_dataset.train_test_split(test_size=0.99999, shuffle=True)
        self.test_data, _ = split['train'], split['test']

        self.test_data = data.filter(lambda x: 'test' in x['metadata'])
        if 'qm9' in args.root:
            representation = 'selfies'
        else:
            representation = 'smiles'
        self.train_dataset = RegressionDataset(self.train_data, self.prompt, representation=representation)
        self.val_dataset = RegressionDataset(self.val_data, self.prompt, representation=representation)
        self.test_dataset = RegressionDataset(self.test_data, self.prompt, representation=representation)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        self.is_gal = args.opt_model.find('galactica') >= 0

    def get_external_data(self, root):
        if 'qm9' in root:
            from datasets import load_dataset

            dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions")
            # property prediction used by mol-instructions dataset is qm9
            qm9_dataset = dataset['property_prediction']
            instructions = INSTRUCTIONS
            if 'homo_lumo_gap' in root:
                target_instruction = instructions['HOMO-LUMO-gap']
                qm9_dataset = qm9_dataset.filter(lambda x: x['instruction'] in target_instruction)
                print('QM9 HOMO-LUMO-gap data preprared')
                print('example of instruction', target_instruction[0])
            elif 'homo' in root:
                target_instruction = instructions['HOMO']
                qm9_dataset = qm9_dataset.filter(lambda x: x['instruction'] in target_instruction)
                print('QM9 HOMO data preprared')
                print('example of instruction', target_instruction[0])
            elif 'lumo' in root:
                target_instruction = instructions['LUMO']
                qm9_dataset = qm9_dataset.filter(lambda x: x['instruction'] in target_instruction)
                print('QM9 LUMO data preprared')
                print('example of instruction', target_instruction[0])
            else:
                raise ValueError
        else:
            raise NotImplementedError
        return qm9_dataset
    
    def classify_property_by_instruciton(self, data):
        unique_instructions = []
        for i in range(len(data)):
            unique_instructions.append(data[i]['instruction'])

        unique_instructions = set(unique_instructions)

        instruction_by_property = {
            'HOMO': [],
            'LUMO': [],
            'HOMO-LUMO-gap': [],
            'etc': []
        }

        for inst in unique_instructions:
            if 'gap' in inst or 'difference' in inst or 'separation' in inst:
                instruction_by_property['HOMO-LUMO-gap'].append(inst)
            elif 'HOMO' in inst:
                instruction_by_property['HOMO'].append(inst)
            elif 'LUMO' in inst:
                instruction_by_property['LUMO'].append(inst)
            else:
                instruction_by_property['etc'].append(inst)

        return instruction_by_property
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        assert self.mode == 'ft'
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.is_gal),
        )
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         drop_last=False,
    #         persistent_workers=True,
    #         collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
    #     )
    #     return [loader,]
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.is_gal),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.is_gal),
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
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.is_gal),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
    