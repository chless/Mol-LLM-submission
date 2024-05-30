# MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter

Codes of our EMNLP2023 paper. [[Paper Link](https://arxiv.org/abs/2310.12798)], [[Website](https://acharkq.github.io/MolCA/)], [[Demo](https://8b8760bb1ba284ef54.gradio.live)]

## Requirements

You can create the environment for MolCA by running the following command in order:

* conda create -n molca python=3.8
* conda activate molca
* conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
* conda install pyg -c pyg
* pip install git+https://github.com/thunlp/OpenDelta.git
* pip install rouge_score nltk ogb peft rdkit salesforce-lavis
* pip install -U transformers pytorch-lightning 
* pip install deepspeed
* Download nltk corpus:

```
import nltk

nltk.download('wordnet')
```

## Dataset

* **PubChem324k**. Download the dataset from [link](https://huggingface.co/datasets/acharkq/PubChem324kV2), and unzip it under the `./data/` directory.
* **CheBI-20, KV-PLM, and MoMu.** Unzip the `./dataset.zip` under the `./data/` directory. 


## Reproduce the results

### Training the Model from Scratch

**Pretrain Stage 1.** Run the following script for stage 1 pretraining on the PubChem324k dataset:

```bash
python stage1.py --root 'data/PubChem324kV2/' --gtm --lm --devices '0,1' --mode train --filename stage1 --rerank_cand_num 128 --num_query_token 8 --tune_gnn
```

**Pretrain Stage 2.** Run the following script for stage 2 pretraining on the PubChem324k dataset:

```bash
python stage2.py --root 'data/PubChem324kV2/' --devices '0,1' --filename "stage2" --stage1_path "all_checkpoints/stage1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --tune_gnn --llm_tune freeze --inference_batch_size 4
```

**Fine-tune Stage.** Run the following script for fine-tuning on the PubChem324k dataset:

```bash
python stage2.py --root 'data/PubChem324kV2/' --devices '0,1' --filename "ft_pubchem324k" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8
```


### Evaluation on Our Pretrained Checkpoints 

We share the checkpoints for reproducing results of molecule-text retrieval and for reproducing results of molecule captioning on the CheBI-20 dataset.

Please download the checkpoints from this [link](https://huggingface.co/acharkq/MolCA/tree/main) and put them under the `./all_checkpoints` directory.

**Molecule-Text Retrieval for PCDes.** Run the following script for evaluation on the PCDes dataset.

```bash
python stage1.py --root 'data/kv_data' --gtm --lm --devices '[0]'  --filename pcdes_evaluation --init_checkpoint "all_checkpoints/share/stage1.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval
```

**Molecule-Text Retrieval for MoMu.** Run the following script for evaluation on the MoMu dataset.

```bash
python stage1.py --root 'data/kv_data' --gtm --lm --devices '[0]'  --filename momu_evaluation --init_checkpoint "all_checkpoints/share/stage1.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval --use_phy_eval
```

**Molecule Captioning.** Run the following script for evaluation on the CheBI-20 dataset.

```bash
python stage2.py --devices '[0]' --filename chebi_evaluation --stage2_path "all_checkpoints/share/chebi.ckpt" --opt_model 'facebook/galactica-1.3b' --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8 --root "data/ChEBI-20_data" --peft_dir "all_checkpoints/share/chebi_lora" --init_checkpoint all_checkpoints/share/chebi.ckpt;
```

## Citation

If you use our codes or checkpoints, please cite our paper:

```bib
@inproceedings{liu2023molca,
    title={MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter},
    author={Liu, Zhiyuan and Li, Sihang and Luo, Yanchen and Fei, Hao and Cao, Yixin and Kawaguchi, Kenji and Wang, Xiang and Chua, Tat-Seng},
    booktitle={EMNLP},
    year={2023},
    url={https://openreview.net/forum?id=14WRhMNq7H}
}
```

# Moleculenet Experiments

## Implemented benchamrks

* PROPERTY_CLASSIFICATION_BENCHMARKS = [
    "bace",  # 1 task # molca, biot5+, instructmol
    "bbbp",  # 1 task # molca, biot5+, instructmol, llasmol
    "clintox",  # 2 tasks # molca, biot5+, llasmol
    "toxcast",  # 617 # molca
    "sider",  # 27 # molca, llasmol
    "tox21",  # 12 tasks # molca
    "hiv",  # 1 tasks # biot5+, instructmol, llasmol
]
* PROPERTY_REGRESSION_BENCHMARKS = [
    "qm9",  # 12 tasks #biot5+, instructmol (homo:2, lumo:3, homo-lumo gap:4)
    "esol",  # 1 task # llasmol
    "lipo",  # 1 task # llasmol
]

```
python MolCA/stage3.py
                "--root=qm9",
                "--subtask_idx=0",
                "--devices", "0",
                "--filename=debugging",
                "--stage2_path=MolCA/all_checkpoints/MolCA/stage2.ckpt",
                "--opt_model=facebook/galactica-1.3b",
                "--mode=ft",
                "--prompt", "[START_I_SMILES]{}[END_I_SMILES].",
                "--tune_gnn",
                "--llm_tune=lora",
                "--inference_batch_size=64",
                "--val_check_interval=0.1",
                "--max_epochs=1",
                "--num_beam=1",
                "--num_random_query_embedding=0",
                "--gnn_jk=layer",
                "--result_file=MolCA/results/molnet_task-specific_0530.csv",
```

* root: task name like qm9
* subtask_idx: subtask idx corresponding to the task
* result_file: a csv file to which evaluation metrics are saved