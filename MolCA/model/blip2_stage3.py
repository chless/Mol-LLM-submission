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
        if args.llm_model.find("galactica") >= 0:
            blip2model = Blip2OPT
        elif args.llm_model.find("llama") >= 0 or args.llm_model.find("vicuna") >= 0:
            blip2model = Blip2Llama
        elif args.llm_model.find("t5") >= 0:
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
        self.num_devices = (
            1
            if isinstance(ast.literal_eval(args.devices), int)
            else len(ast.literal_eval(args.devices))
        )
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

        if isinstance(batch, list):
            #TODO: IMPORTANT!! get tie batch index and key specified in config
            batch_sizes = self.args.batch_sizes
            total_batch_size = sum(batch_sizes.values())

            batch_partisions = list(self.args.batch_sizes.keys())
            outputs = {}
            for i in range(len(batch)):
                outputs[batch_partisions[i]] = self.blip2model(batch[i])

            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                batch_size=total_batch_size,
                sync_dist=False,
            )
            # log dataset specific losses
            for i, key in enumerate(outputs):
                task_subtask_pairs = batch[i][0].task_subtask_pair
                instance_losses = outputs[key]["instance_loss"]

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

            total_loss = sum(
                [
                    outputs[key]["loss"] * batch_sizes[key] / total_batch_size
                    for key in outputs.keys()
                ]
            )

            self.log(
                "train/total_loss",
                float(total_loss),
                batch_size=total_batch_size,
                sync_dist=False,
            )
            self.log(
                "train_total_loss",
                float(total_loss),
                batch_size=total_batch_size,
                sync_dist=False,
            )
            return total_loss
        else:
            raise NotImplementedError()

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

    def evaluation_in_train_step(self, batch, predictions, logits):
        graphs, prompt_tokens, texts, tasks = batch

        targets = self.blip2model.llm_tokenizer.batch_decode(texts.input_ids)
        prompts = self.blip2model.llm_tokenizer.batch_decode(
            prompt_tokens.input_ids, skip_special_tokens=False
        )
        predictions = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in predictions
        ]
        targets = [
            t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in targets
        ]
        prompts = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in prompts
        ]
        probs = convert_logit2binary_prob(logits, self.blip2model.llm_tokenizer)

        self.train_list_predictions.append(predictions)
        self.train_list_targets.append(targets)
        self.train_list_prompts.append(prompts)
        self.train_list_tasks.append(tasks)
        self.train_list_probs.append(probs)

    # not use, because evaluatino and logging in training step make x4 times per epoch training time
    def on_train_evaluation_end(self, mode="train"):
        print("on_evaluation_epoch_end start")
        list_predictions = self.train_list_predictions
        list_targets = self.train_list_targets
        list_tasks = self.train_list_tasks
        list_probs = self.train_list_probs
        list_prompts = self.train_list_prompts

        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        tasks = [i for ii in list_tasks for i in ii]
        probs = [i for ii in list_probs for i in ii]
        prompts = [i for ii in list_prompts for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_tasks = [None for _ in range(self.trainer.world_size)]
        all_probs = [None for _ in range(self.trainer.world_size)]
        all_prompts = [None for _ in range(self.trainer.world_size)]

        if self.num_devices > 1:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_tasks, tasks)
            dist.all_gather_object(all_probs, probs)
            dist.all_gather_object(all_prompts, prompts)
        else:
            all_predictions[0] = predictions
            all_targets[0] = targets
            all_tasks[0] = tasks
            all_probs[0] = probs
            all_prompts[0] = prompts

        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            all_tasks = [i for ii in all_tasks for i in ii]
            all_probs = [i for ii in all_probs for i in ii]
            all_prompts = [i for ii in all_prompts for i in ii]
            self.save_predictions(
                predictions=all_predictions,
                targets=all_targets,
                tasks=all_tasks,
                prompts=all_prompts,
                filename=f"{self.args.mode}-step{self.global_step}_predictions.json",
            )

            evaluation_results, failed_cases = task_specifically_evaluate(
                predictions=all_predictions,
                targets=all_targets,
                tasks=all_tasks,
                probs=all_probs,
                prompts=all_prompts,
                tokenizer=self.blip2model.llm_tokenizer,
            )

            self.save_predictions(
                predictions=failed_cases["predictions"],
                targets=failed_cases["targets"],
                tasks=failed_cases["tasks"],
                prompts=failed_cases["prompts"],
                filename=f"{self.args.mode}-step{self.global_step}_failed_cases.json",
            )

            for task_subtask_pair in evaluation_results:
                for metric in evaluation_results[task_subtask_pair]:
                    self.log(
                        f"{mode}/{task_subtask_pair}/{metric}",
                        evaluation_results[task_subtask_pair][metric],
                        sync_dist=False,
                    )

        # reset the lists
        self.train_list_predictions = []
        self.train_list_targets = []
        self.train_list_prompts = []
        self.train_list_tasks = []
        self.train_list_probs = []

    def on_evaluation_epoch_start(self):
        self.list_predictions = []
        self.list_targets = []
        self.list_prompts = []
        self.list_tasks = []
        self.list_probs = []
        self.total_avg_loss = 0.0
        self.total_seen_data_size = 0
        self.eval_dataset_losses = {}

    def evaluation_step(self, batch, batch_idx, dataloader_idx, mode="val"):
        graphs, prompt_tokens, texts = batch
        tasks = graphs.task_subtask_pair

        samples = {"graphs": graphs, "prompt_tokens": prompt_tokens}
        outputs = self.blip2model.generate(
            samples,
            num_beams=self.num_beams,
            max_length=self.gen_max_len,
            min_length=self.min_len,
        )
        predictions = outputs.predictions
        targets = self.blip2model.llm_tokenizer.batch_decode(texts.input_ids)
        prompts = self.blip2model.llm_tokenizer.batch_decode(
            prompt_tokens.input_ids, skip_special_tokens=False
        )
        predictions = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in predictions
        ]
        targets = [
            t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in targets
        ]
        prompts = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in prompts
        ]
        probs = convert_logit2binary_prob(outputs.logits, self.blip2model.llm_tokenizer)

        self.list_predictions.append(predictions)
        self.list_targets.append(targets)
        self.list_prompts.append(prompts)
        self.list_tasks.append(tasks)
        self.list_probs.append(probs)

        batch_size = texts.input_ids.shape[0]
        outputs = self.blip2model(batch)  # omit tasks when inputting to the model
        ##============== Overall Loss ===================##

        new_data_weight = batch_size / (self.total_seen_data_size + batch_size)
        self.total_avg_loss += (
            outputs["loss"].item() - self.total_avg_loss
        ) * new_data_weight
        self.total_seen_data_size += batch_size

        task_subtask_pairs = tasks
        instance_losses = outputs["instance_loss"]

        for task_subtask_pair in task_subtask_pairs:
            if task_subtask_pair not in self.eval_dataset_losses.keys():
                self.eval_dataset_losses[task_subtask_pair] = {
                    "avg_loss": 0.0,
                    "total_samples": 0,
                }

        for i in range(instance_losses.shape[0]):
            task_subtask_pair = task_subtask_pairs[i]
            # calculate average loss
            self.eval_dataset_losses[task_subtask_pair][
                "avg_loss"
            ] *= self.eval_dataset_losses[task_subtask_pair]["total_samples"] / (
                self.eval_dataset_losses[task_subtask_pair]["total_samples"] + 1
            )
            self.eval_dataset_losses[task_subtask_pair]["avg_loss"] += instance_losses[
                i
            ] / (self.eval_dataset_losses[task_subtask_pair]["total_samples"] + 1)

            self.eval_dataset_losses[task_subtask_pair]["total_samples"] += 1

        return outputs["loss"]

    def on_evaluation_epoch_end(self, mode="val") -> None:
        print("on_evaluation_epoch_end start")
        list_predictions = self.list_predictions
        list_targets = self.list_targets
        list_tasks = self.list_tasks
        list_probs = self.list_probs
        list_prompts = self.list_prompts

        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        tasks = [i for ii in list_tasks for i in ii]
        probs = [i for ii in list_probs for i in ii]
        prompts = [i for ii in list_prompts for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_tasks = [None for _ in range(self.trainer.world_size)]
        all_probs = [None for _ in range(self.trainer.world_size)]
        all_prompts = [None for _ in range(self.trainer.world_size)]

        if self.num_devices > 1:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_tasks, tasks)
            dist.all_gather_object(all_probs, probs)
            dist.all_gather_object(all_prompts, prompts)
        else:
            all_predictions[0] = predictions
            all_targets[0] = targets
            all_tasks[0] = tasks
            all_probs[0] = probs
            all_prompts[0] = prompts

        if self.global_rank == 0:
            self.log(f"{mode}/total_loss", self.total_avg_loss, sync_dist=False)

            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            all_tasks = [i for ii in all_tasks for i in ii]
            all_probs = [i for ii in all_probs for i in ii]
            all_prompts = [i for ii in all_prompts for i in ii]
            self.save_predictions(
                predictions=all_predictions,
                targets=all_targets,
                tasks=all_tasks,
                prompts=all_prompts,
                filename=(
                    f"{self.args.mode}-step{self.global_step}_predictions.json"
                    if self.args.mode == "val"
                    else f"{self.args.mode}_predictions.json"
                ),
            )

            evaluation_results, failed_cases = task_specifically_evaluate(
                predictions=all_predictions,
                targets=all_targets,
                tasks=all_tasks,
                probs=all_probs,
                prompts=all_prompts,
                tokenizer=self.blip2model.llm_tokenizer,
            )

            self.save_predictions(
                predictions=failed_cases["predictions"],
                targets=failed_cases["targets"],
                tasks=failed_cases["tasks"],
                prompts=failed_cases["prompts"],
                filename=(
                    f"{self.args.mode}-step{self.global_step}_failed_cases.json"
                    if self.args.mode == "val"
                    else f"{self.args.mode}_failed_cases.json"
                ),
            )

            for task_subtask_pair in evaluation_results:
                for metric in evaluation_results[task_subtask_pair]:
                    self.log(
                        f"{mode}/{task_subtask_pair}/{metric}",
                        evaluation_results[task_subtask_pair][metric],
                        sync_dist=False,
                    )

            for dataset in self.eval_dataset_losses.keys():
                self.log(
                    f"{mode}/{dataset}/avg_loss",
                    self.eval_dataset_losses[dataset]["avg_loss"],
                    batch_size=self.eval_dataset_losses[dataset]["total_samples"],
                    sync_dist=False,
                )
