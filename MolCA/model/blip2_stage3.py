import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
from model.blip2_t5 import Blip2T5
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import (
    task_specifically_evaluate,
    AttrDict,
    convert_logit2binary_prob,
)
from transformers import Adafactor
import ast
import json
from data_provider.stage3_dm import (
    TOTAL_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    CLASSIFICATION_BENCHMARKS,
    REACTION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
)


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    # try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1 :]
            if key == "":
                return value
            module_state_dict[key] = value
    return module_state_dict


class Blip2Stage3(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint["state_dict"].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint["state_dict"].pop(key)

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, "do_sample"):
            args.do_sample = False
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.gen_max_len = args.gen_max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        if args.opt_model.find("galactica") >= 0:
            self.blip2opt = Blip2OPT(
                args.bert_name,
                args.gin_num_layers,
                args.gin_hidden_dim,
                args.drop_ratio,
                args.tune_gnn,
                args.num_query_token,
                args.cross_attention_freq,
                args.llm_tune,
                args.peft_dir,
                args.opt_model,
                args.prompt,
                args,
            )
        elif args.opt_model.find("llama") >= 0 or args.opt_model.find("vicuna") >= 0:
            self.blip2opt = Blip2Llama(
                args.bert_name,
                args.gin_num_layers,
                args.gin_hidden_dim,
                args.drop_ratio,
                args.tune_gnn,
                args.num_query_token,
                args.cross_attention_freq,
                args.llm_tune,
                args.peft_dir,
                args.opt_model,
                args.prompt,
                args,
            )
        elif args.opt_model.find("t5") >= 0:
            self.blip2opt = Blip2T5(
                args.bert_name,
                args.gin_num_layers,
                args.gin_hidden_dim,
                args.drop_ratio,
                args.tune_gnn,
                args.num_query_token,
                args.cross_attention_freq,
                args.llm_tune,
                args.peft_dir,
                args.opt_model,
                args.prompt,
                args,
            )
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.num_devices = (
            1
            if isinstance(ast.literal_eval(args.devices), int)
            else len(ast.literal_eval(args.devices))
        )
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        graph_encoder_dict = get_module_state_dict(
            state_dict, "blip2qformer.graph_encoder"
        )
        qformer_dict = get_module_state_dict(state_dict, "blip2qformer.Qformer")
        ln_graph_dict = get_module_state_dict(state_dict, "blip2qformer.ln_graph")
        qs_weight = get_module_state_dict(state_dict, "blip2qformer.query_tokens")
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self

    def configure_optimizers(self):
        if self.args.optimizer == "adafactor":
            print("Using adafactor optimizer")
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = min(
                len(self.trainer.train_dataloader), self.args.warmup_steps
            )
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
            if self.args.scheduler == "linear_warmup_cosine_lr":
                self.scheduler = LinearWarmupCosineLRScheduler(
                    optimizer,
                    self.args.max_epochs,
                    self.args.min_lr,
                    self.args.init_lr,
                    warmup_steps,
                    self.args.warmup_lr,
                )
            elif self.args.scheduler == "linear_warmup_step_lr":
                self.scheduler = LinearWarmupStepLRScheduler(
                    optimizer,
                    self.args.max_epochs,
                    self.args.min_lr,
                    self.args.init_lr,
                    self.args.lr_decay_rate,
                    self.args.warmup_lr,
                    warmup_steps,
                )
            elif self.args.scheduler == "None":
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def save_predictions(self, predictions, targets, tasks):
        assert len(predictions) == len(targets)
        assert len(predictions) == len(tasks)
        instances = []
        for i in range(len(predictions)):
            instances.append(
                {
                    "task": tasks[i],
                    "prediction": predictions[i],
                    "target": targets[i],
                }
            )
        with open(os.path.join(self.logger.log_dir, "predictions.json"), "w") as f:
            """
            for i in range(len(predictions)):
                line = {
                    "task": tasks[i],
                    "prediction": predictions[i],
                    "target": targets[i],
                }
                f.write(json.dumps(line, ensure_ascii=True) + "\n")
            """
            json.dump(instances, f, ensure_ascii=False, indent=4)

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="test")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(mode="test")

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(mode="val")

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        if isinstance(batch, list) and len(batch) == 5:
            batch_sizes = [b[1].input_ids.size(0) for b in batch]
            batch_size_dict = {
                "classification": batch_sizes[0],
                "regression": batch_sizes[1],
                "reaction": batch_sizes[2],
                "reagent": batch_sizes[3],
                "translation": batch_sizes[4],
            }
            total_batch_size = sum(batch_sizes)
            ##============== Overall Loss ===================##
            (
                classification_batch,
                regression_batch,
                reaction_batch,
                reagent_batch,
                translation_batch,
            ) = batch
            losses = {
                "classification": self.blip2opt(classification_batch[:-1]),
                "regression": self.blip2opt(regression_batch[:-1], task="regression"),
                "reaction": self.blip2opt(reaction_batch[:-1]),
                "reagent": self.blip2opt(reagent_batch[:-1]),
                "translation": self.blip2opt(translation_batch[:-1]),
            }
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                batch_size=total_batch_size,
                sync_dist=True,
            )
            for key, loss in losses.items():
                self.log(
                    f"{key}_loss",
                    float(loss["loss"]),
                    batch_size=batch_size_dict[key],
                    sync_dist=True,
                )
            # TODO: refactor hardcoded loss scale
            total_loss = (
                losses["classification"]["loss"] * batch_sizes[0]
                + losses["regression"]["loss"] * batch_sizes[1]
                + losses["reaction"]["loss"] * batch_sizes[2]
                + losses["reagent"]["loss"] * batch_sizes[3]
                + losses["translation"]["loss"] * batch_sizes[4]
            ) / sum(batch_sizes)
            self.log(
                "total_loss",
                float(total_loss),
                batch_size=total_batch_size,
                sync_dist=True,
            )

            return total_loss
        else:
            batch_size = batch[1].input_ids.size(0)  #
            ##============== Overall Loss ===================##
            loss = self.blip2opt(batch)
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                batch_size=batch_size,
                sync_dist=True,
            )
            for key, loss_item in loss.items():
                self.log(key, float(loss_item), batch_size=batch_size, sync_dist=True)
            return loss["loss"]

    def on_train_epoch_end(self) -> None:
        if self.args.llava_style:
            max_epoch = self.args.max_epochs
            current_epoch = self.trainer.current_epoch
            if current_epoch >= (max_epoch // 2 - 1):
                for name, param in self.blip2opt.opt_model.named_parameters():
                    name_split = name.split(".")
                    if len(name_split) > 3:
                        if name_split[-3] == "lora_A" or name_split[-3] == "lora_B":
                            param.requires_grad = True
                print("set lora_A and lora_B to True for next epoch")

    def on_evaluation_epoch_start(self):
        self.list_predictions = []
        self.list_targets = []
        self.list_tasks = []
        self.list_probs = []
        self.total_avg_loss = 0.0
        self.total_seen_data_size = 0
        self.batch_losses = []
        self.dict_task_losses = {}

    def evaluation_step(self, batch, batch_idx, dataloader_idx, mode="val"):
        if dataloader_idx == 0:
            task = "classification"
        elif dataloader_idx == 1:
            task = "regression"
        elif dataloader_idx == 2:
            task = "reaction"
        elif dataloader_idx == 3:
            task = "reagent"
        elif dataloader_idx == 4:
            task = "translation"
        # TODO: figure out why batch composition is different from training_step
        graphs, prompt_tokens, texts, tasks = batch

        samples = {"graphs": graphs, "prompt_tokens": prompt_tokens}
        outputs = self.blip2opt.generate(
            samples,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_length=self.gen_max_len,
            min_length=self.min_len,
        )
        predictions = outputs.predictions
        targets = self.blip2opt.opt_tokenizer.batch_decode(texts.input_ids)
        self.list_predictions.append(predictions)
        self.list_targets.append(targets)
        self.list_tasks.append(tasks)
        # TODO: implement exception for tasks other than classification
        probs = convert_logit2binary_prob(outputs.logits, self.blip2opt.opt_tokenizer)
        self.list_probs.append(probs)

        batch_size = texts.input_ids.shape[0]
        loss = self.blip2opt(batch[:-1])  # omit tasks when inputting to the model
        ##============== Overall Loss ===================##
        for key, loss_item in loss.items():
            self.log(
                f"{mode}/{task}",
                float(loss_item),
                batch_size=batch_size,
                sync_dist=True,
            )

        new_data_weight = batch_size / (self.total_seen_data_size + batch_size)
        self.total_avg_loss += (
            loss["loss"].item() - self.total_avg_loss
        ) * new_data_weight
        self.total_seen_data_size += batch_size

        return loss["loss"]

    def on_evaluation_epoch_end(self, mode="val") -> None:
        print("on_evaluation_epoch_end start")
        list_predictions = self.list_predictions
        list_targets = self.list_targets
        list_tasks = self.list_tasks
        list_probs = self.list_probs

        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        tasks = [i for ii in list_tasks for i in ii]
        probs = [i for ii in list_probs for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_tasks = [None for _ in range(self.trainer.world_size)]
        all_probs = [None for _ in range(self.trainer.world_size)]

        if self.num_devices > 1:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_tasks, tasks)
            dist.all_gather_object(all_probs, probs)
        else:
            all_predictions[0] = predictions
            all_targets[0] = targets
            all_tasks[0] = tasks
            all_probs[0] = probs

        if self.global_rank == 0:
            self.log(f"{mode}/total_loss", self.total_avg_loss, sync_dist=False)

            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            all_tasks = [i for ii in all_tasks for i in ii]
            all_probs = [i for ii in all_probs for i in ii]
            self.save_predictions(all_predictions, all_targets, all_tasks)

            evaluation_results = task_specifically_evaluate(
                predictions=all_predictions,
                targets=all_targets,
                tasks=all_tasks,
                probs=all_probs,
                tokenizer=self.blip2opt.opt_tokenizer,
                text_trunc_length=self.gen_max_len * 2,
            )
            for task_subtask_pair in evaluation_results:
                for metric in evaluation_results[task_subtask_pair]:
                    self.log(
                        f"{mode}/{task_subtask_pair}/{metric}",
                        evaluation_results[task_subtask_pair][metric],
                        sync_dist=False,
                    )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument("--gin_hidden_dim", type=int, default=300)
        parser.add_argument("--gin_num_layers", type=int, default=5)
        parser.add_argument("--drop_ratio", type=float, default=0.0)
        parser.add_argument("--tune_gnn", action="store_true", default=False)
        # Bert
        parser.add_argument("--bert_hidden_dim", type=int, default=768, help="")
        parser.add_argument("--bert_name", type=str, default="scibert")
        parser.add_argument("--cross_attention_freq", type=int, default=2)
        parser.add_argument("--num_query_token", type=int, default=8)
        # OPT
        parser.add_argument("--opt_model", type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument("--num_beams", type=int, default=5)
        parser.add_argument("--do_sample", action="store_true", default=False)
        parser.add_argument("--gen_max_len", type=int, default=256)
        parser.add_argument("--min_len", type=int, default=8)
        parser.add_argument("--llm_tune", type=str, default="freeze")
        parser.add_argument("--peft_config", type=str, default=None)
        parser.add_argument("--peft_dir", type=str, default="")

        parser.add_argument("--save_every_n_epochs", type=int, default=10)
        # quantization
        parser.add_argument("--load_in_8bit", action="store_true", default=False)

        # lora config
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=32)
        parser.add_argument("--lora_dropout", type=int, default=0.1)

        # optimization
        parser.add_argument("--reaction_weight", type=float, default=1.0)
        parser.add_argument(
            "--weight_decay", type=float, default=0.05, help="optimizer weight decay"
        )
        parser.add_argument(
            "--init_lr", type=float, default=1e-4, help="optimizer init learning rate"
        )
        parser.add_argument(
            "--min_lr", type=float, default=1e-5, help="optimizer min learning rate"
        )
        parser.add_argument(
            "--warmup_lr",
            type=float,
            default=1e-6,
            help="optimizer warmup learning rate",
        )
        parser.add_argument(
            "--warmup_steps", type=int, default=1000, help="optimizer warmup steps"
        )
        parser.add_argument(
            "--lr_decay_rate", type=float, default=0.9, help="optimizer lr decay rate"
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="linear_warmup_cosine_lr",
            help="type of scheduler",
        )  # or linear_warmup_step_lr
        parser.add_argument(
            "--optimizer", type=str, default="adamw", help="type of scheduler"
        )
        parser.add_argument("--stage1_path", type=str, default="")
        parser.add_argument("--stage2_path", type=str, default="")
        parser.add_argument("--init_checkpoint", type=str, default="")
        parser.add_argument("--graph_decoder_ckpt", type=str, default=None)
        parser.add_argument("--coeff_recon_loss", type=float, default=0.2)

        parser.add_argument("--used_gnn_layer", type=int, default=-1)
        parser.add_argument("--gnn_jk", type=str, default="layer")
        parser.add_argument("--num_random_query_embedding", type=int, default=0)
        parser.add_argument(
            "--mol_representation",
            type=str,
            default="string+graph",
            choices=["string_only", "graph_only", "string+graph"],
        )
        parser.add_argument("--add_reg_tokens", type=bool, default=True)
        parser.add_argument(
            "--apply_reg_order_scale", action="store_true", default=False
        )
        parser.add_argument("--apply_reg_label_quant", type=int, default=-1)
        parser.add_argument("--add_selfies_tokens", action="store_true", default=True)
        parser.add_argument(
            "--selfies_token_path", type=str, default="MolCA/model/selfies_dict.txt"
        )
        return parent_parser
