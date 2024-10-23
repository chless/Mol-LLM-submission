import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
from model.blip2_mistral import Blip2Mistral
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
from transformers.utils import logging

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

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
        self.num_beams = args.num_beams
        self.gen_max_len = args.gen_max_len
        self.min_len = args.min_len
        self.tune_llm = args.tune_llm
        self.on_second_stage = False
        # set strict_loading to False to load model in a lightweight way
        self.strict_loading = False
        if "galactica" in args.llm_model:
            blip2model = Blip2OPT
        elif "llama" in args.llm_model:
            blip2model = Blip2Llama
        elif "mistral" in args.llm_model:
            blip2model = Blip2Mistral
        elif "t5" in args.llm_model:
            blip2model = Blip2T5
        else:
            raise NotImplementedError()

        self.blip2model = blip2model(
            args.bert_name,
            args.gin_num_layers,
            args.gin_hidden_dim,
            args.drop_ratio,
            args.tune_gnn,
            args.num_query_token,
            args.cross_attention_freq,
            args.tune_llm,
            args.peft_dir,
            args.llm_model,
            args.prompt,
            args,
        )
        self.tokenizer = self.blip2model.init_tokenizer()
        self.num_moving_samples = 32
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
        load_ignore_unexpected(self.blip2model.Qformer, qformer_dict)
        self.blip2model.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2model.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2model.query_tokens.data.copy_(qs_weight)
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

    def save_predictions(
        self, predictions, targets, tasks, prompts, filename="predictions.json"
    ):
        assert len(predictions) == len(targets)
        assert len(predictions) == len(tasks)
        assert len(predictions) == len(prompts)
        instances = []
        for i in range(len(predictions)):
            instances.append(
                {
                    "task": tasks[i],
                    "prediction": predictions[i],
                    "target": targets[i],
                    "prompt": prompts[i],
                }
            )
        os.makedirs(self.logger.log_dir, exist_ok=True)

        with open(os.path.join(self.logger.log_dir, filename), "w") as f:
            json.dump(instances, f, ensure_ascii=False, indent=4)

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="test")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(mode="test")

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(mode="val")

    def apply_separated_stage(self):
        if (
            self.trainer.global_step >= self.args.second_stage_start_step
            and not self.on_second_stage
        ):
            self.blip2model.set_params_requires_grads(
                model=self.blip2model.llm_model,
                keyword="lora",
                grad=True,
                IsPrint=False,
            )
            self.on_second_stage = True
            print("set lora weights trainable")

    def training_step(self, batch, batch_idx):
        if self.args.llava_style:
            self.apply_separated_stage()

        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = self.args.batch_size

        outputs = self.blip2model(batch)

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            batch_size=batch_size,
            sync_dist=False,
        )
        # log dataset specific losses
        task_subtask_pairs = batch[0].task_subtask_pair

        # TODO: implement for sequence packing
        if isinstance(task_subtask_pairs[0], list):
            # unwrap the list of list
            task_subtask_pairs = [i for ii in task_subtask_pairs for i in ii]
        else:
            instance_losses = outputs["instance_loss"]

            for task_subtask_pair in task_subtask_pairs:
                if task_subtask_pair not in self.dataset_losses.keys():
                    self.dataset_losses[task_subtask_pair] = []

            for i in range(instance_losses.shape[0]):
                task_subtask_pair = task_subtask_pairs[i]
                # calculate average loss
                self.dataset_losses[task_subtask_pair].append(instance_losses[i])

                while (
                    len(self.dataset_losses[task_subtask_pair])
                    > self.num_moving_samples
                ):
                    self.dataset_losses[task_subtask_pair].pop(0)

            for dataset in self.dataset_losses.keys():
                self.log(
                    f"train/{dataset}/loss",
                    sum(self.dataset_losses[dataset])
                    / len(self.dataset_losses[dataset]),
                    batch_size=len(self.dataset_losses[dataset]),
                    sync_dist=False,
                )

        loss = outputs["loss"]
        self.log(
            "train_total_loss",
            float(loss),
            batch_size=batch_size,
            sync_dist=False,
        )
        return loss

    def on_train_epoch_start(self) -> None:
        if self.blip2model.llm_tokenizer.mol_string_randomization_ratio > 0:
            # conduct mol_string_randomization_ratio annealing, so that at max epochs, it is 0
            self.blip2model.llm_tokenizer.mol_string_randomization_ratio = max(
                0,
                self.blip2model.llm_tokenizer.mol_string_randomization_ratio
                * (1 - self.trainer.current_epoch / self.trainer.max_epochs),
            )

        self.log(
            "mol_string_randomization_ratio",
            self.blip2model.llm_tokenizer.mol_string_randomization_ratio,
            sync_dist=False,
        )

        self.dataset_losses = {}

        self.train_list_predictions = []
        self.train_list_targets = []
        self.train_list_prompts = []
        self.train_list_tasks = []
        self.train_list_probs = []
        self.train_total_avg_loss = 0.0
        self.train_total_seen_data_size = 0

    def on_evaluation_epoch_start(self):
        self.list_logs = {
            "predictions": [],
            "targets": [],
            "tasks": [],
            "probs": [],
            "prompts": [],
        }

        self.total_avg_loss = 0.0
        self.total_seen_data_size = 0
        self.task_subtask_name_pairs = self.trainer.datamodule.dataset_split['test'].task_subtask_name_pairs
        self.eval_dataset_losses = {
            task_subtask_pair: {"avg_loss": 0.0, "num_instances": 0}
            for task_subtask_pair in self.task_subtask_name_pairs
        }

    def evaluation_step(self, batch, batch_idx, dataloader_idx, mode="val"):
        graphs, prompt_tokens, target_tokens = batch

        samples = {"graphs": graphs, "input_tokens": prompt_tokens}
        outputs = self.blip2model.generate(
            samples,
            num_beams=self.num_beams,
            max_length=self.gen_max_len,
            min_length=self.min_len,
        )
        predictions = outputs.predictions
        targets = self.blip2model.llm_tokenizer.batch_decode(target_tokens.input_ids)
        prompts = self.blip2model.llm_tokenizer.batch_decode(
            prompt_tokens.input_ids, skip_special_tokens=False
        )

        predictions = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in predictions
        ]
        targets = [
            t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in targets
        ]
        tasks = graphs.task_subtask_pair
        probs = convert_logit2binary_prob(outputs.logits, self.blip2model.llm_tokenizer)
        prompts = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in prompts
        ]

        # TODO: calculate evaluation metric incide a process, and average using count from task_subtask_pair
        # TODO: save predictions incide a process asynchrously

        self.list_logs["predictions"].extend(predictions)
        self.list_logs["targets"].extend(targets)
        self.list_logs["tasks"].extend(tasks)
        self.list_logs["probs"].extend(probs)
        self.list_logs["prompts"].extend(prompts)

        batch_size = prompt_tokens.input_ids.shape[0]
        outputs = self.blip2model(batch)  # omit tasks when inputting to the model
        ##============== Overall Loss ===================##

        new_data_weight = batch_size / (self.total_seen_data_size + batch_size)
        self.total_avg_loss += (outputs["loss"] - self.total_avg_loss) * new_data_weight
        self.total_seen_data_size += batch_size

        task_subtask_pairs = tasks
        instance_losses = outputs["instance_loss"]

        for i in range(instance_losses.shape[0]):
            task_subtask_pair = task_subtask_pairs[i]
            # calculate average loss
            self.eval_dataset_losses[task_subtask_pair][
                "avg_loss"
            ] *= self.eval_dataset_losses[task_subtask_pair]["num_instances"] / (
                self.eval_dataset_losses[task_subtask_pair]["num_instances"] + 1
            )
            self.eval_dataset_losses[task_subtask_pair]["avg_loss"] += instance_losses[
                i
            ] / (self.eval_dataset_losses[task_subtask_pair]["num_instances"] + 1)

            self.eval_dataset_losses[task_subtask_pair]["num_instances"] += 1

        return outputs["loss"]

    def on_evaluation_epoch_end(self, mode="val") -> None:
        print(f"\nDevice {self.device} on_evaluation_epoch_end start")

        evaluation_results, failed_cases = task_specifically_evaluate(
            predictions=self.list_logs["predictions"],
            targets=self.list_logs["targets"],
            tasks=self.list_logs["tasks"],
            prompts=self.list_logs["prompts"],
            probs=self.list_logs["probs"],
            tokenizer=self.blip2model.llm_tokenizer,
            total_task_subtask_pairs=self.task_subtask_name_pairs,
        )

        self.save_predictions(
            predictions=self.list_logs["predictions"],
            targets=self.list_logs["targets"],
            tasks=self.list_logs["tasks"],
            prompts=self.list_logs["prompts"],
            filename=(
                f"{self.args.mode}-step{self.global_step}-{self.global_rank}-outputs.json"
                if self.args.mode == "val"
                else f"{self.args.mode}-{self.global_rank}-outputs.json"
            ),
        )

        self.save_predictions(
            predictions=failed_cases["predictions"],
            targets=failed_cases["targets"],
            tasks=failed_cases["tasks"],
            prompts=failed_cases["prompts"],
            filename=(
                f"{self.args.mode}-step{self.global_step}-{self.global_rank}-failed_cases.json"
                if self.args.mode == "val"
                else f"{self.args.mode}-{self.global_rank}-failed_cases.json"
            ),
        )

        self.log(
            f"{mode}/total_loss",
            self.total_avg_loss,
            sync_dist=True,
            batch_size=self.total_seen_data_size,
        )
        flattened_metric_keys = []
        flattened_metric_tensors = torch.empty(size=(0,2), device=self.device)

        # tied to order of self.task_subtask_name_pairs
        for task_subtask_pair in evaluation_results:
            for metric in evaluation_results[task_subtask_pair]:
                flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}")
                metric_value = evaluation_results[task_subtask_pair][metric]
                num_instance = evaluation_results[task_subtask_pair]["num_instances"]
                metric_count_pair = [
                    metric_value * num_instance, 
                    num_instance
                    ]
                
                flattened_metric_tensors = torch.cat(
                    [
                        flattened_metric_tensors,
                        torch.tensor(
                            metric_count_pair,
                            device=self.device,
                        ).unsqueeze(0)
                    ],
                    dim=0
                )

        # tied to order of self.task_subtask_name_pairs
        for dataset in self.eval_dataset_losses.keys():
            flattened_metric_keys.append(f"{mode}/{dataset}/avg_loss")
            metric_value = self.eval_dataset_losses[dataset]["avg_loss"]
            num_instance = self.eval_dataset_losses[dataset]["num_instances"]
            metric_count_pair = [
                metric_value * num_instance, 
                num_instance
                ]
            flattened_metric_tensors = torch.cat(
                [
                    flattened_metric_tensors,
                    torch.tensor(
                        metric_count_pair,
                        device=self.device,
                    ).unsqueeze(0)
                ],
                dim=0
            )
        
        assert flattened_metric_tensors.shape[0] == len(flattened_metric_keys), f"flattened_metric_tensors.shape[0]: {flattened_metric_tensors.shape[0]}, len(flattened_metric_keys): {len(flattened_metric_keys)}"        
        if self.trainer.world_size > 1:
            print("gather the metrics across devices")
            gathered_flattened_metric_tensors = self.all_gather(flattened_metric_tensors) # [world_size, num_metrics, metric_value * per_device_instance_count, per_device_instance_count]
            print("metrics are gathered, {}".format(gathered_flattened_metric_tensors.shape))
            summed_flattened_metric_tensors = gathered_flattened_metric_tensors[:, :, 0].sum(dim=0)
            total_instance_count = gathered_flattened_metric_tensors[:, :, 1].sum(dim=0)
        else:
            summed_flattened_metric_tensors = flattened_metric_tensors[:, 0]
            total_instance_count = flattened_metric_tensors[:, 1]

        
        # if total_instance_count is 0, set the metric to null value
        averaged_flattened_metric_tensors = torch.where(
            total_instance_count > 0,
            summed_flattened_metric_tensors / total_instance_count,
            torch.tensor(float("nan"), device=self.device),
        )

        print(averaged_flattened_metric_tensors)
        if self.trainer.is_global_zero:
            print("============================== Evaluation Results ==============================")
            for i, key in enumerate(flattened_metric_keys):
                print(f"{key}: {averaged_flattened_metric_tensors[i]}")
                self.log(
                    key,
                    averaged_flattened_metric_tensors[i],
                    sync_dist=False,
                    batch_size=int(total_instance_count[i]),
                )
            print("=================================================================================")
            

        print(f"\nDevice {self.device} on_evaluation_epoch_end end")
