import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, NeptuneLogger, TensorBoardLogger
from data_provider.stage3_dm import (
    Stage3DM,
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
)
from data_provider.stage2_chebi_dm import Stage2CheBIDM
from model.blip2_stage3 import Blip2Stage3

# instruction-tuning for benchmark datasets

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
# for A5000 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)


class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)


def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.task is None:
        model = Blip2Stage3
    else:
        raise NotImplementedError()

    # decide model initialization
    if args.init_checkpoint:
        model = model.load_from_checkpoint(
            args.init_checkpoint, strict=False, args=args
        )
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = model(args)
        ckpt = torch.load(args.stage2_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = model(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = model(args)

    print("total params:", sum(p.numel() for p in model.parameters()))

    if args.opt_model.find("galactica") >= 0 or args.opt_model.find("t5") >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find("llama") >= 0 or args.opt_model.find("vicuna") >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    import ast

    devices = ast.literal_eval(args.devices)
    num_devices = len(devices) if not isinstance(devices, int) else 1
    # adjust intended total batch size is the same regarlless of the number of devices
    adjustBatchSize(args, num_devices)

    if args.root.lower().find("chebi") >= 0:
        dm = Stage2CheBIDM(
            args.mode,
            args.num_workers,
            args.batch_size,
            args.root,
            args.text_max_len,
            tokenizer,
            args,
        )
    else:
        dm = Stage3DM(
            args.mode,
            args.num_workers,
            args.batch_size,
            args.root,
            args.text_max_len,
            tokenizer,
            args,
        )

    # callbacks to save model parameters
    callbacks = []
    if args.not_save_model:
        pass
    else:
        monitoring_metric = "total_loss"
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(args.checkpoint_save_dir, args.filename),
                filename="{step:05d}-{total_loss:.3f}",
                every_n_train_steps=args.every_n_train_steps,
                save_last=True,
                save_top_k=10,
                save_on_train_epoch_end=True,
                monitor=monitoring_metric,
                mode="min",
            )
        )
        callbacks.append(
            SaveLoRAModelCallback(f"MolCA/all_checkpoints/{args.filename}/lora")
        )

    if len(args.devices.split(",")) > 1:
        if args.strategy_name == "fsdp":
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == "deepspeed":
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPStrategy(find_unused_parameters=True, start_method="spawn")
    else:
        strategy = "auto"
        args.devices = [eval(args.devices)]
    # logger setting
    logger = CSVLogger(save_dir=f"./MolCA/all_checkpoints/{args.filename}/")
    """
    neptune_logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project=args.neptune_project,
        log_model_checkpoints=False
    )
    """
    tb_logger = TensorBoardLogger(
        f"./MolCA/all_checkpoints/tensorboard/",
        name=args.result_file.split("/")[-1].split(".")[0],
    )

    trainer_args = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "callbacks": callbacks,
        "strategy": strategy,
        "logger": [logger, tb_logger],
    }
    if args.max_steps > 0:
        trainer_args["max_steps"] = args.max_steps
    else:
        trainer_args["max_epochs"] = args.max_epochs
    trainer = Trainer(**trainer_args)
    if args.mode in {"pretrain", "ft", "multi_task"}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
        outputs = trainer.test(model, datamodule=dm)

    # TODO: Deprecate eval mode.
    # Previously, molca authors evaluate validation dataset and testset at the same in validation epoch.
    # Now, we separate validation and testset evaluation, as usual.
    elif args.mode == "eval":
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    elif args.mode == "test":
        outputs = trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()

    if args.result_file is not None and args.root != "multi_task":
        update_result_csv(
            args=args, outputs=outputs, task_names=dm.train_data.get_task_names()
        )

def adjustBatchSize(args, num_devices):
    if args.root == "multi_task":
        assert args.batch_size % 4 == 0, "batch size should be multiple of 4"
        assert args.inference_batch_size % 4 == 0, "batch size should be multiple of 4"
        args.batch_size = args.batch_size // 4
        args.inference_batch_size = args.inference_batch_size // 4
        
    assert (
        args.batch_size % num_devices == 0
    ), "batch size should be multiple of num_devices"
    assert (
        args.inference_batch_size % num_devices == 0
    ), "batch size should be multiple of num_devices"
    args.batch_size = args.batch_size // num_devices
    args.inference_batch_size = args.inference_batch_size // num_devices
    print(f"batch size per device: {args.batch_size}")
    print(f"inference batch size per device: {args.inference_batch_size}")


def update_result_csv(args, outputs, task_names):
    # first, read the content in result_csv file
    import json

    single_tasks = (
        REGRESSION_BENCHMARKS
        + CLASSIFICATION_BENCHMARKS
        + MOL2TEXT_BENCHMARKS
        + TEXT2MOL_BENCHMARKS
    )
    if args.root in single_tasks:
        if os.path.exists(args.result_file):
            results_dict = json.load(open(args.result_file, "r"))
        else:
            os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
            results_dict = dict()

        subtask = task_names[args.subtask_idx]
        # TODO: extend multiple dataloader, for multi-task intruction-tuning
        # currently, len(outputs)=1
        for idx in range(len(outputs)):
            output = outputs[idx]
            for k in output.keys():
                # extend results_dict with new key
                if args.root not in results_dict:
                    results_dict[args.root] = dict()
                if subtask not in results_dict[args.root]:
                    results_dict[args.root][subtask] = dict()
                if k not in results_dict[args.root][subtask]:
                    results_dict[args.root][subtask][k] = output[k]
        # finally, save the updated results_dict
        with open(args.result_file, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Updated the result file {args.result_file}")
    elif args.root == "multi_task":
        with open(args.result_file, "w") as f:
            json.dump(outputs, f, indent=4)
    else:
        raise NotImplementedError()


class SaveLoRAModelCallback(Callback):
    def __init__(self, dir_path):
        """
        Args:
            save_path (str): Path where the model and LoRA weights should be saved.
        """
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def on_epoch_end(self, trainer, pl_module):
        """
        Called when an epoch ends.
        """
        model = (
            pl_module.model
        )  # Assuming the LoRA adapted model is stored in this property
        model.save_pretrained(self.dir_path)
        print(
            f"Peft lora weights saved to {self.dir_path} at epoch {trainer.current_epoch}"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="stage2_test")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # MM settings
    parser.add_argument("--mode", type=str, default="pretrain")
    parser.add_argument("--strategy_name", type=str, default=None)
    parser.add_argument("--iupac_prediction", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage3.add_model_specific_args(parser)  # add model args
    parser = Stage3DM.add_model_specific_args(parser)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--every_n_train_steps", type=int, default=1000)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=0.1)
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--neptune_project", type=str, default="chless/text-mol")
    parser.add_argument("--result_file", type=str, default="MolCA/results/debug.json")
    parser.add_argument("--not_save_model", action="store_true", default=False)
    parser.add_argument("--checkpoint_save_dir", type=str, default="MolCA/all_checkpoints/")

    # added args
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    return args


if __name__ == "__main__":
    main(get_args())
