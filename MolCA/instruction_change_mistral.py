import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from data_provider.stage3_dm import (
    Stage3DM,
    Mol_LLM_Dataset,
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    TOTAL_BENCHMARKS,
)

from model.blip2_stage3 import Blip2Stage3
import json
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
from tqdm import tqdm

from data_provider.instructions_new import (
    FS, RS, REAGENT,
    NC_S2F, NC_S2I, NC_I2S, NC_I2F,
    MC, MG,
    HOMO, LUMO, HOMO_LUMO_GAP, ESOL, LIPO,
    BACE, BBBP, CLINTOX_FDA, CLINTOX_CT,
    TOXCAST, TOX21, HIV, 
    QM9_MU, QM9_ALPHA, QM9_R2, QM9_ZPVE, QM9_CV, QM9_U298, QM9_H298, QM9_G298,
)
import random
SYSTEM = "You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation. </s></s>"

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", message=r".*Skipped loading .*")
warnings.filterwarnings("ignore", message=r".*No normalization for .*")
# for A5000 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)


class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint, strict=False):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)


@hydra.main(config_path="configs", config_name="default.yaml", version_base=None)
def main(cfg):
    cfg = flatten_dictconfig(cfg)
    pl.seed_everything(cfg.seed)

    model = Blip2Stage3

    # decide model initialization
    if cfg.stage2_path:
        model = model(cfg)
        ckpt = torch.load(cfg.stage2_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded stage2 model from {cfg.stage2_path}")
    else:
        model = model(cfg)

    print("total params:", sum(p.numel() for p in model.parameters()))

    dm = Stage3DM(
        mode=cfg.mode,
        num_workers=cfg.num_workers,
        tokenizer=model.blip2model.llm_tokenizer,
        fit_llm_input_convention=model.blip2model.fit_llm_input_convention,
        fit_llm_output_convention=model.blip2model.fit_llm_output_convention,
        args=cfg,
    )

    train_data = dm.train_dataloader().dataset
    max_len = len(train_data)
    tokenizer = model.blip2model.llm_tokenizer
    processed_data_list = []
    sider_train_cnt = 0
    for i in tqdm(range(max_len), desc=f"Change for train data"):
        dt = train_data[i]
        label = dt["task_subtask_pair"]
        if(label == 'smol-forward_synthesis/smol-forward_synthesis' or label == 'forward_reaction_prediction/forward_reaction_prediction'):
            new_inst_part = random.choice(FS)
        elif(label == 'smol-retrosynthesis/smol-retrosynthesis' or label == 'retrosynthesis/retrosynthesis'):
            new_inst_part = random.choice(RS)
        elif(label == 'reagent_prediction/reagent_prediction'):
            new_inst_part = random.choice(REAGENT)
            in_out_txt = dt["input_text"]
            out_txt = in_out_txt.split("[/INST]")[-1]
            mol_sub = in_out_txt.split("<SELFIES>",1)[-1]
            mol = mol_sub.split("</SELFIES> [/INST]")[0]
            mol = mol.replace(";",".",10)
            new_in_txt = new_inst_part.replace("<INPUT>","<SELFIES>"+mol+"</SELFIES>")
            prompt_tokens = tokenizer("<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]", return_length=True)
            if isinstance(prompt_tokens.length, list):
                prompt_tokens_length = prompt_tokens.length[0]
            elif isinstance(prompt_tokens.length, int):
                prompt_tokens_length = prompt_tokens.length
            new_target = prompt_tokens_length * tokenizer.pad_token + out_txt
            dt["input_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"+out_txt
            dt["target_text"] = new_target
            processed_data_list.append(dt)
            continue
        elif('name_conversion' in label):
            if(label == 'smol-name_conversion-s2f/smol-name_conversion-s2f'):
                new_inst_part = random.choice(NC_S2F)
            elif(label == 'smol-name_conversion-s2i/smol-name_conversion-s2i'):
                new_inst_part = random.choice(NC_S2I)
            elif(label =='smol-name_conversion-i2s/smol-name_conversion-i2s'):
                new_inst_part = random.choice(NC_I2S)
            elif(label == 'smol-name_conversion-i2f/smol-name_conversion-i2f'):
                new_inst_part = random.choice(NC_I2F)
            else:
                print("No label exist: Error.")
        elif('qm9' in label):
            if(label == 'qm9_homo/qm9_homo'): 
                new_inst_part = random.choice(HOMO)
            elif(label == 'qm9_lumo/qm9_lumo'):  
                new_inst_part = random.choice(LUMO)
            elif(label == 'qm9_homo_lumo_gap/qm9_homo_lumo_gap'):
                new_inst_part = random.choice(HOMO_LUMO_GAP)
            elif(label == 'qm9_additional_label/mu'):
                new_inst_part = random.choice(QM9_MU)
            elif(label == 'qm9_additional_label/alpha'):
                new_inst_part = random.choice(QM9_ALPHA)
            elif(label == 'qm9_additional_label/r2'): 
                new_inst_part = random.choice(QM9_R2)
            elif(label == 'qm9_additional_label/zpve'): 
                new_inst_part = random.choice(QM9_ZPVE)
            elif(label == 'qm9_additional_label/cv'): 
                new_inst_part = random.choice(QM9_CV)
            elif(label == 'qm9_additional_label/u298'): 
                new_inst_part = random.choice(QM9_U298)
            elif(label == 'qm9_additional_label/h298'):
                new_inst_part = random.choice(QM9_H298)
            elif(label == 'qm9_additional_label/g298'):
                new_inst_part = random.choice(QM9_G298)
            else:
                print("No label exist: Error.")
        elif(label == 'smol-molecule_captioning/smol-molecule_captioning' or label == 'chebi-20-mol2text/chebi-20-mol2text'):
            new_inst_part = random.choice(MC)
        elif(label == 'bace/Class'):
            new_inst_part = random.choice(BACE)
        elif(label == 'bbbp/p_np'):
            new_inst_part = random.choice(BBBP)
        elif(label == 'clintox/FDA_APPROVED'):
            new_inst_part = random.choice(CLINTOX_FDA)
        elif(label == 'clintox/CT_TOX'):
            new_inst_part = random.choice(CLINTOX_CT)
        elif(label == 'toxcast/ACEA_T47D_80hr_Negative'):
            new_inst_part = random.choice(TOXCAST)
        elif(label == 'sider/Hepatobiliary disorders'):
            sider_train_cnt+=1
            continue
        elif(label == 'tox21/NR-AR'):
            new_inst_part = random.choice(TOX21)
        elif(label == 'hiv/HIV_active'): 
            new_inst_part = random.choice(HIV)
        elif(label == 'esol/measured log solubility in mols per litre'):
            new_inst_part = random.choice(ESOL)
        elif(label == 'lipo/exp'):
            new_inst_part = random.choice(LIPO)
        elif(label == 'smol-molecule_generation/smol-molecule_generation' or label == 'chebi-20-text2mol/chebi-20-text2mol'):
            new_inst_part = random.choice(MG)
            in_out_txt = dt["input_text"]
            out_txt = in_out_txt.split("[/INST]")[-1]
            desc_sub = in_out_txt.split("<DESCRIPTION>")[-1]
            desc = desc_sub.split("</DESCRIPTION>")[0]
            new_in_txt = new_inst_part.replace("<INPUT>","<DESCRIPTION>"+desc+"</DESCRIPTION>")
            prompt_tokens = tokenizer("<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]", return_length=True)
            if isinstance(prompt_tokens.length, list):
                prompt_tokens_length = prompt_tokens.length[0]
            elif isinstance(prompt_tokens.length, int):
                prompt_tokens_length = prompt_tokens.length
            new_target = prompt_tokens_length * tokenizer.pad_token + out_txt
            dt["input_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"+out_txt
            dt["target_text"] = new_target
            processed_data_list.append(dt)
            continue
        else:
            print("No label exist: Error.")
        in_out_txt = dt["input_text"]
        out_txt = in_out_txt.split("[/INST]")[-1]
        mol_sub = in_out_txt.split("<SELFIES>",1)[-1]
        mol = mol_sub.split("</SELFIES>",1)[0]
        mol = mol.replace(";",".",10)
        new_in_txt = new_inst_part.replace("<INPUT>","<SELFIES>"+mol+"</SELFIES>")
        prompt_tokens = tokenizer("<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]", return_length=True)
        if isinstance(prompt_tokens.length, list):
            prompt_tokens_length = prompt_tokens.length[0]
        elif isinstance(prompt_tokens.length, int):
            prompt_tokens_length = prompt_tokens.length
        new_target = prompt_tokens_length * tokenizer.pad_token + out_txt
        dt["input_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"+out_txt
        dt["target_text"] = new_target
        processed_data_list.append(dt)
                
    assert len(processed_data_list)==(max_len-sider_train_cnt)
    Mol_LLM_Dataset.save(
        processed_data_list,
        "/home/dydrkfl0608/new_inst_mistral_data_train.pt"
    )
    
    test_data = dm.test_dataloader().dataset
    test_len = len(test_data)
    processed_test_list = []
    sider_test_cnt = 0
    for i in tqdm(range(test_len), desc=f"Change for test data"):
        dt = test_data[i]
        label = dt["task_subtask_pair"]
        if(label == 'smol-forward_synthesis/smol-forward_synthesis' or label == 'forward_reaction_prediction/forward_reaction_prediction'):
            new_inst_part = random.choice(FS)
        elif(label == 'smol-retrosynthesis/smol-retrosynthesis' or label == 'retrosynthesis/retrosynthesis'):
            new_inst_part = random.choice(RS)
        elif(label == 'reagent_prediction/reagent_prediction'):
            new_inst_part = random.choice(REAGENT)
            in_out_txt = dt["input_text"]
            out_txt = in_out_txt.split("[/INST]")[-1]
            mol_sub = in_out_txt.split("<SELFIES>",1)[-1]
            mol = mol_sub.split("</SELFIES> [/INST]")[0]
            mol = mol.replace(";",".",10)
            new_in_txt = new_inst_part.replace("<INPUT>","<SELFIES>"+mol+"</SELFIES>")
            prompt_tokens = tokenizer("<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]", return_length=True)
            if isinstance(prompt_tokens.length, list):
                prompt_tokens_length = prompt_tokens.length[0]
            elif isinstance(prompt_tokens.length, int):
                prompt_tokens_length = prompt_tokens.length
            new_target = prompt_tokens_length * tokenizer.pad_token + out_txt
            dt["input_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"+out_txt
            dt["target_text"] = new_target
            dt["prompt_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"
            processed_test_list.append(dt)
            continue
        elif('name_conversion' in label):
            if(label == 'smol-name_conversion-s2f/smol-name_conversion-s2f'):
                new_inst_part = random.choice(NC_S2F)
            elif(label == 'smol-name_conversion-s2i/smol-name_conversion-s2i'):
                new_inst_part = random.choice(NC_S2I)
            elif(label =='smol-name_conversion-i2s/smol-name_conversion-i2s'):
                new_inst_part = random.choice(NC_I2S)
            elif(label == 'smol-name_conversion-i2f/smol-name_conversion-i2f'):
                new_inst_part = random.choice(NC_I2F)
            else:
                print("No label exist: Error.")
        elif('qm9' in label):
            if(label == 'qm9_homo/qm9_homo'): 
                new_inst_part = random.choice(HOMO)
            elif(label == 'qm9_lumo/qm9_lumo'):  
                new_inst_part = random.choice(LUMO)
            elif(label == 'qm9_homo_lumo_gap/qm9_homo_lumo_gap'):
                new_inst_part = random.choice(HOMO_LUMO_GAP)
            elif(label == 'qm9_additional_label/mu'):
                new_inst_part = random.choice(QM9_MU)
            elif(label == 'qm9_additional_label/alpha'):
                new_inst_part = random.choice(QM9_ALPHA)
            elif(label == 'qm9_additional_label/r2'): 
                new_inst_part = random.choice(QM9_R2)
            elif(label == 'qm9_additional_label/zpve'): 
                new_inst_part = random.choice(QM9_ZPVE)
            elif(label == 'qm9_additional_label/cv'): 
                new_inst_part = random.choice(QM9_CV)
            elif(label == 'qm9_additional_label/u298'): 
                new_inst_part = random.choice(QM9_U298)
            elif(label == 'qm9_additional_label/h298'):
                new_inst_part = random.choice(QM9_H298)
            elif(label == 'qm9_additional_label/g298'):
                new_inst_part = random.choice(QM9_G298)
            else:
                print("No label exist: Error.")
        elif(label == 'smol-molecule_captioning/smol-molecule_captioning' or label == 'chebi-20-mol2text/chebi-20-mol2text'):
            new_inst_part = random.choice(MC)
        elif(label == 'bace/Class'):
            new_inst_part = random.choice(BACE)
        elif(label == 'bbbp/p_np'):
            new_inst_part = random.choice(BBBP)
        elif(label == 'clintox/FDA_APPROVED'):
            new_inst_part = random.choice(CLINTOX_FDA)
        elif(label == 'clintox/CT_TOX'):
            new_inst_part = random.choice(CLINTOX_CT)
        elif(label == 'toxcast/ACEA_T47D_80hr_Negative'):
            new_inst_part = random.choice(TOXCAST)
        elif(label == 'sider/Hepatobiliary disorders'):
            sider_test_cnt+=1
            continue
        elif(label == 'tox21/NR-AR'):
            new_inst_part = random.choice(TOX21)
        elif(label == 'hiv/HIV_active'): 
            new_inst_part = random.choice(HIV)
        elif(label == 'esol/measured log solubility in mols per litre'):
            new_inst_part = random.choice(ESOL)
        elif(label == 'lipo/exp'):
            new_inst_part = random.choice(LIPO)
        elif(label == 'smol-molecule_generation/smol-molecule_generation' or label == 'chebi-20-text2mol/chebi-20-text2mol'):
            new_inst_part = random.choice(MG)
            in_out_txt = dt["input_text"]
            out_txt = in_out_txt.split("[/INST]")[-1]
            desc_sub = in_out_txt.split("<DESCRIPTION>")[-1]
            desc = desc_sub.split("</DESCRIPTION>")[0]
            new_in_txt = new_inst_part.replace("<INPUT>","<DESCRIPTION>"+desc+"</DESCRIPTION>")
            prompt_tokens = tokenizer("<s> [INST] "+ SYSTEM + new_in_txt+" [/INST]", return_length=True)
            if isinstance(prompt_tokens.length, list):
                prompt_tokens_length = prompt_tokens.length[0]
            elif isinstance(prompt_tokens.length, int):
                prompt_tokens_length = prompt_tokens.length
            new_target = prompt_tokens_length * tokenizer.pad_token + out_txt
            dt["input_text"] = "<s> [INST] "+ SYSTEM + new_in_txt+" [/INST]"+out_txt
            dt["target_text"] = new_target
            dt["prompt_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"
            processed_test_list.append(dt)
            continue
        else:
            print("No label exist: Error.")
        in_out_txt = dt["input_text"]
        out_txt = in_out_txt.split("[/INST]")[-1]
        mol_sub = in_out_txt.split("<SELFIES>",1)[-1]
        mol = mol_sub.split("</SELFIES>",1)[0]
        mol = mol.replace(";",".",10)
        new_in_txt = new_inst_part.replace("<INPUT>","<SELFIES>"+mol+"</SELFIES>")
        prompt_tokens = tokenizer("<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]", return_length=True)
        if isinstance(prompt_tokens.length, list):
            prompt_tokens_length = prompt_tokens.length[0]
        elif isinstance(prompt_tokens.length, int):
            prompt_tokens_length = prompt_tokens.length
        new_target = prompt_tokens_length * tokenizer.pad_token + out_txt
        dt["input_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"+out_txt
        dt["target_text"] = new_target
        dt["prompt_text"] = "<s> [INST] "+ SYSTEM +new_in_txt+" [/INST]"
        processed_test_list.append(dt)
                
    assert len(processed_test_list)==(test_len-sider_test_cnt)
    Mol_LLM_Dataset.save(
        processed_test_list,
        "/home/dydrkfl0608/new_inst_mistral_data_test.pt"
    )
    
    # # callbacks to save model parameters
    # callbacks = []

    # monitoring_metric = "train_total_loss"
    # callbacks.append(
    #     ModelCheckpoint(
    #         dirpath=os.path.join(cfg.logging_dir, cfg.filename),
    #         filename="{step:05d}-{train_total_loss:.3f}",
    #         every_n_train_steps=cfg.every_n_train_steps,
    #         save_last=True,
    #         save_top_k=5,
    #         save_on_train_epoch_end=True,
    #         monitor=monitoring_metric,
    #         mode="min",
    #     )
    # )

    # if len(cfg.devices.split(",")) > 1:
    #     if cfg.strategy_name == "fsdp":
    #         strategy = strategies.DDPFullyShardedNativeStrategy()
    #     elif cfg.strategy_name == "deepspeed":
    #         strategy = strategies.DeepSpeedStrategy(stage=3)
    #     else:
    #         strategy = MyDDPStrategy(find_unused_parameters=False, start_method="spawn")
    # else:
    #     strategy = "auto"
    #     cfg.devices = [eval(cfg.devices)]

    # logger = CSVLogger(save_dir=os.path.join(cfg.logging_dir, cfg.filename))

    # wandb_logger = WandbLogger(
    #     name=cfg.filename,
    #     project=cfg.wandb_project,
    #     entity=cfg.wandb_entity,
    #     id=cfg.wandb_id,
    # )

    # tb_logger = TensorBoardLogger(
    #     os.path.join(cfg.logging_dir, "tensorboard"),
    #     name=cfg.filename,
    # )

    # trainer_args = {
    #     "accelerator": cfg.accelerator,
    #     "devices": cfg.devices,
    #     "precision": cfg.precision,
    #     "callbacks": callbacks,
    #     "strategy": strategy,
    #     "logger": [logger, wandb_logger, tb_logger],
    #     "max_steps": cfg.max_steps,
    #     "val_check_interval": cfg.val_check_interval,
    #     "check_val_every_n_epoch": cfg.check_val_every_n_epoch,
    #     "accumulate_grad_batches": cfg.accumulate_grad_batches,
    # }

    # if cfg.skip_sanity_check:
    #     trainer_args["num_sanity_val_steps"] = 0
    # if hasattr(cfg, "profiler"):
    #     trainer_args["profiler"] = cfg.profiler

    # trainer = Trainer(**trainer_args)
    # if cfg.mode in {"pretrain", "ft", "multi_task"}:
    #     trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt_path)
    #     outputs = trainer.test(model, datamodule=dm)

    # elif cfg.mode == "eval":
    #     trainer.fit_loop.epoch_progress.current.completed = cfg.caption_eval_epoch - 1
    #     trainer.validate(model, datamodule=dm)
    # elif cfg.mode == "test":
    #     outputs = trainer.test(model, datamodule=dm)
    # else:
    #     raise NotImplementedError()

    # if cfg.filename is not None:
    #     update_result_csv(
    #         logger_dir=trainer.logger.log_dir,
    #         outputs=outputs,
    #     )


def update_result_csv(outputs, logger_dir, task_names=None):
    # first, read the content in result_csv file
    single_tasks = (
        REGRESSION_BENCHMARKS
        + CLASSIFICATION_BENCHMARKS
        + MOL2TEXT_BENCHMARKS
        + TEXT2MOL_BENCHMARKS
    )
    performance_result_path = os.path.join(logger_dir, "benchmark_performance.json")

    os.makedirs(os.path.dirname(performance_result_path), exist_ok=True)
    # task average of output
    final_output = dict()
    tasks = [list(o.keys()) for o in outputs]
    tasks = list(set([item for sublist in tasks for item in sublist]))

    for task in tasks:
        final_output[task] = []
        for output in outputs:
            if task in output:
                final_output[task].append(output[task])
        final_output[task] = final_output[task][
            0
        ]  # identical, just replicated 4 dataloader

    with open(performance_result_path, "w") as f:
        json.dump(final_output, f, indent=4)


def flatten_dictconfig(config: DictConfig) -> DictConfig:
    """
    Flatten a nested DictConfig into a single level DictConfig with keys as the path to the original keys.

    Args:
    - config (DictConfig): The nested DictConfig to be flattened.
    - parent_key (str, optional): The base key to use for prefixing the keys. Defaults to ''.
    - separator (str, optional): The separator to use between keys. Defaults to '.'.

    Returns:
    - DictConfig: The flattened configuration.
    """

    # only flatten just first level
    items = []
    for k, v in config.items():
        new_key = k
        if isinstance(v, DictConfig):
            for kk, vv in v.items():
                items.append((f"{kk}", vv))
        else:
            items.append((new_key, v))
    return OmegaConf.create(dict(items))


if __name__ == "__main__":
    main()
