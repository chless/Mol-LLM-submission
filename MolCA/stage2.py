import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.stage2_dm import Stage2DM
from data_provider.iupac_dm import IupacDM
from data_provider.stage2_chebi_dm import Stage2CheBIDM
from data_provider.stage2_regression_dm import Stage2RegressionDM
from model.blip2_stage2 import Blip2Stage2
from model.blip2_regression import Blip2Regression

import neptune
from pytorch_lightning.loggers import NeptuneLogger

# torch.set_default_dtype(torch.float16)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
# for A5000 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)

# strategy = strategies.DDPStrategy(find_unused_parameters=find_unused_parameters, start_method='spawn')
# class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
#     def load_model_state_dict(self, checkpoint):
#         assert self.lightning_module is not None
#         self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)


class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)


def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.task is None:
        model = Blip2Stage2
    elif "regression" in args.task:
        model = Blip2Regression
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
    args.batch_size = args.batch_size // num_devices
    if args.task == "iupac_prediction":
        dm = IupacDM(
            args.mode,
            args.num_workers,
            args.batch_size,
            args.root,
            args.text_max_len,
            tokenizer,
            args,
        )
    else:
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
        elif args.root.lower().find("qm9") >= 0:
            dm = Stage2RegressionDM(
                args.mode,
                args.num_workers,
                args.batch_size,
                args.root,
                args.text_max_len,
                tokenizer,
                args,
            )
        else:
            dm = Stage2DM(
                args.mode,
                args.num_workers,
                args.batch_size,
                args.root,
                args.text_max_len,
                tokenizer,
                args,
            )

    callbacks = []
    # fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="MolCA/all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(
        plc.ModelCheckpoint(
            dirpath="MolCA/all_checkpoints/" + args.filename + "/",
            filename="{epoch:02d}",
            every_n_epochs=args.save_every_n_epochs,
            save_last=True,
            save_top_k=-1,
            save_on_train_epoch_end=True,
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
    neptune_logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="chanhui-lee/text-mol",
    )

    trainer_args = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "callbacks": callbacks,
        "strategy": strategy,
        "logger": [logger, neptune_logger],
    }
    if args.max_steps > 0:
        trainer_args["max_steps"] = args.max_steps
    else:
        trainer_args["max_epochs"] = args.max_epochs
    trainer = Trainer(**trainer_args)
    if args.mode in {"pretrain", "ft"}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
        # test after training
        output = trainer.test(model, datamodule=dm)
    elif args.mode == "eval":
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    elif args.mode == "test":
        output = trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


class SaveLoRAModelCallback(plc.Callback):
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
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2DM.add_model_specific_args(parser)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=0.1)

    # added args
    parser.add_argument(
        "--graph_embedding_mse_logging", action="store_true", default=False
    )
    parser.add_argument(
        "--graph_embedding_mse_backprop", action="store_true", default=False
    )
    parser.add_argument("--qformer_instruction", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    return args


if __name__ == "__main__":
    main(get_args())
