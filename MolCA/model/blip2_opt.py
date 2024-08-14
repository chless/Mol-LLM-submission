"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    TaskType,
    PeftModel,
)
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
import model.added_tokens as added_tokens
from data_provider.stage3_dm import CUSTOM_SEQ_RE

# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]


def mask_by_len(input, lens, fill_value=0):
    """
    input: shape = [N, D]
    lens: shape = [N]
    """
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph["node_feat"])
    edge_index = torch.from_numpy(
        graph["edge_index"],
    )
    edge_attr = torch.from_numpy(graph["edge_feat"])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


import re


class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune="freeze",
        peft_dir="",
        opt_model="facebook/galactica-1.3b",
        prompt="",  # TODO: remove. currently LLM classes not use prompt from args.prompt
        args=None,
    ):
        super().__init__()
        self.args = args
        self.peft_dir = peft_dir

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(
            gin_num_layers, gin_hidden_dim, gin_drop_ratio, args
        )

        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            bert_name,
            num_query_token,
            gin_hidden_dim,
            cross_attention_freq,
            bert_num_hidden_layers=args.bert_num_hidden_layers,
        )

        ## remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # initialize opt model
        self.opt_tokenizer = AutoTokenizer.from_pretrained(
            opt_model, use_fast=False, padding_side="right"
        )
        self.opt_tokenizer.mol_string_randomization_ratio = (
            args.mol_string_randomization_ratio
        )
        self.add_necessary_tokens()

        self.collater = Collater([], [])

        if opt_model == "facebook/galactica-125m":
            self.opt_model = OPTForCausalLM.from_pretrained(
                opt_model, torch_dtype=torch.bfloat16
            )
        else:
            if torch.cuda.is_bf16_supported():
                self.opt_model = OPTForCausalLM.from_pretrained(
                    opt_model, torch_dtype=torch.bfloat16
                )
            else:
                self.opt_model = OPTForCausalLM.from_pretrained(
                    opt_model, torch_dtype=torch.float16
                )
        self.opt_model.resize_token_embeddings(
            len(self.opt_tokenizer)
        )  # this will cause bug when full fine-tuning the opt model

        self.llm_tune = llm_tune
        if llm_tune == "lora":
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(
                    self.opt_model, peft_dir, is_trainable=True
                )
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(
                        **LoraConfig.from_json_file(self.args.peft_config)
                    )
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                    )
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
                self.opt_model.print_trainable_parameters()
        elif llm_tune == "freeze":
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == "full":
            pass
        else:
            raise NotImplementedError()

        # fixme: this is different from the original BLIP2
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        for name, param in self.opt_model.named_parameters():
            name_split = name.split(".")
            if name_split[-2] == "embed_tokens" or name_split[-2] == "embed_positions":
                param.requires_grad = True
        print("set embed_tokens and embed_positions to trainable")

        if self.args.llava_style:
            for name, param in self.opt_model.named_parameters():
                name_split = name.split(".")
                if len(name_split) > 3:
                    if name_split[-3] == "lora_A" or name_split[-3] == "lora_B":
                        param.requires_grad = False
            print("set lora_A and lora_B to non-trainable")

    def add_necessary_tokens(self):
        self.opt_tokenizer.add_special_tokens({"pad_token": "<pad>"})

        if self.args.add_selfies_tokens:
            # Read txt from selfies_token_path
            with open(self.args.selfies_token_path, "r") as f:
                selfies_tokens = f.readlines()
                selfies_tokens = [token.strip() for token in selfies_tokens]
            self.opt_tokenizer.add_tokens(selfies_tokens)
            # get token id of the selfies_tokens
            self.opt_tokenizer.selfies_token_ids = [
                self.opt_tokenizer(token, add_special_tokens=False).input_ids[0]
                for token in selfies_tokens
            ]
            self.opt_tokenizer.added_selfies_tokens = selfies_tokens
            # remove '.' from the marked list for selfies token
            self.opt_tokenizer.added_selfies_tokens.remove(".")
            self.opt_tokenizer.selfies_token_ids.remove(36)
            print(f"Added {len(selfies_tokens)} selfies tokens to the tokenizer")

        additional_tokens = [
            getattr(added_tokens, tokens)
            for tokens in dir(added_tokens)
            if not re.match("__.*__", tokens)
        ]
        additional_tokens = [
            token for sublist in additional_tokens for token in sublist
        ]
        self.opt_tokenizer.add_tokens(additional_tokens)

        self.mol_token = added_tokens.MOL_EMBEDDING[0]
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer(
            self.mol_token, add_special_tokens=False
        ).input_ids[0]

    def merge_and_initialize_lora(self):
        self.model.blip2opt.opt_model.merge_and_unload(progressbar=True)

        if self.llm_tune == "lora":
            if self.peft_dir:
                self.opt_model = PeftModel.from_pretrained(
                    self.opt_model, self.peft_dir, is_trainable=True
                )
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(
                        **LoraConfig.from_json_file(self.args.peft_config)
                    )
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.args.lora_r,
                        lora_alpha=self.args.lora_alpha,
                        lora_dropout=self.args.lora_dropout,
                    )
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
                self.opt_model.print_trainable_parameters()
        elif self.llm_tune == "freeze":
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif self.llm_tune == "full":
            pass
        else:
            raise NotImplementedError()

    def random_replace_mol_string(self, prompt_tokens_input_ids):
        ids = prompt_tokens_input_ids
        tokenizer = self.opt_tokenizer
        mol_string_randomization_ratio = tokenizer.mol_string_randomization_ratio
        total_selfies_token_ids = tokenizer.selfies_token_ids

        selfies_min_id = min(total_selfies_token_ids)
        selfies_max_id = max(total_selfies_token_ids)
        # if ids are correspond to total_selfies_token_ids, replace them with random token by mol_string_randomization_ratio
        full_random_replaced = torch.where(
            (ids >= selfies_min_id) & (ids <= selfies_max_id),
            torch.randint(
                selfies_min_id, selfies_max_id + 1, ids.shape, device=ids.device
            ),
            ids,
        )
        partial_random_replaced = torch.where(
            torch.rand(ids.shape, device=ids.device) < mol_string_randomization_ratio,
            full_random_replaced,
            ids,
        )
        return partial_random_replaced

    def forward(self, batch, task=None):
        # graph, smiles tokens, molecule description tokens
        graphs, prompt_tokens, text_tokens = batch
        device = prompt_tokens.input_ids.device

        if self.args.mol_string_randomization_ratio > 0:
            prompt_tokens.input_ids = self.random_replace_mol_string(
                prompt_tokens.input_ids
            )

        empty_targets = (
            torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long)
            .to(device)
            .fill_(-100)
        )
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        # Prompt_embeds takes 139 tokens, but the model only takes 8 tokens.
        # Though we use original setting of MolCA, this is unecessary context length comsumption.
        if "graph" in self.args.mol_representation:
            self.inject_graph_embeds2prompt_embeds(
                prompt_embeds=prompt_embeds,
                prompt_tokens=prompt_tokens,
                graphs=graphs,
            )

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        attention_mask = torch.cat(
            [prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1
        )

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        results = dict()
        if self.args.apply_reg_order_scale and task == "regression":
            logits = outputs.logits
            # calculate ce loss using logits and targets
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            ce_loss = ce_loss.view(targets.size())
            order_scale = torch.ones(size=ce_loss.shape, device=ce_loss.device)
            order_scale[
                :, self.args.prompt_max_len : self.args.prompt_max_len + 5
            ] *= 10
            order_scale[
                :, self.args.prompt_max_len + 5 : self.args.prompt_max_len + 6
            ] *= 5
            ce_loss = ce_loss * order_scale
            ce_loss_gathered_by_non_pad = ce_loss[targets != -100].sum()
            loss = ce_loss_gathered_by_non_pad / (targets != -100).sum()

        else:
            loss = outputs.loss

        results.update({"loss": loss})
        return results

    def inject_graph_embeds2prompt_embeds(self, prompt_embeds, prompt_tokens, graphs):
        if "reactant_x" in graphs.keys():
            mol_token_sequence = []
            for mol in ["reactant", "product"]:
                mol_x = graphs[f"{mol}_x"]
                mol_edge_index = graphs[f"{mol}_edge_index"]
                mol_edge_attr = graphs[f"{mol}_edge_attr"]
                mol_batch = graphs[f"{mol}_batch"]
                mol_embeds, mol_masks = self.graph_encoder(
                    mol_x, mol_edge_index, mol_edge_attr, mol_batch
                )
                if not self.tune_gnn:
                    mol_embeds = mol_embeds.detach()
                mol_embeds = self.ln_graph(mol_embeds, mol_masks)
                query_tokens = self.query_tokens.expand(mol_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=mol_embeds,
                    encoder_attention_mask=mol_masks,
                    return_dict=True,
                )
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
                mol_token_sequence.append(mol_tokens)
            mol_tokens = torch.cat(mol_token_sequence, dim=1)
            for i in range(prompt_tokens.is_mol_token.shape[0]):
                # only inject mol tokens to the prompt embeds when there is mol token in the prompt
                if prompt_embeds[i][prompt_tokens.is_mol_token[i]].shape[0]:
                    # there are cases that mol token is truncated, which make error in vectorized operation
                    for j in range(
                        prompt_embeds[i][prompt_tokens.is_mol_token[i]].shape[0]
                    ):
                        prompt_embeds[i][prompt_tokens.is_mol_token[i]][j] = mol_tokens[
                            i
                        ][j]

            # prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

        else:
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            if not self.tune_gnn:
                graph_embeds = graph_embeds.detach()
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks,  # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state)
            for i in range(prompt_tokens.is_mol_token.shape[0]):
                # only inject mol tokens to the prompt embeds when there is mol token in the prompt
                if prompt_embeds[i][prompt_tokens.is_mol_token[i]].shape[0]:
                    # there are cases that mol token is truncated, which make error in vectorized operation
                    for j in range(
                        prompt_embeds[i][prompt_tokens.is_mol_token[i]].shape[0]
                    ):
                        prompt_embeds[i][prompt_tokens.is_mol_token[i]][j] = mol_tokens[
                            i
                        ][j]
        return prompt_embeds

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples["graphs"]
        prompt_tokens = samples["prompt_tokens"]
        # prompt_lens = samples['prompt_lens']
        # with self.maybe_autocast():

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        if "graph" in self.args.mol_representation:
            self.inject_graph_embeds2prompt_embeds(
                prompt_embeds=prompt_embeds,
                prompt_tokens=prompt_tokens,
                graphs=graphs,
            )

        outputs = self.opt_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            # min_length=min_length,
            min_new_tokens=min_length,  # TODO: change to min_new_tokens for all layered methods
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
        )

        # TODO solve the minor discrepancy between logits decoded and output sequence decoded
        scores = outputs.scores
        batch_size, sequence_length = outputs.sequences.shape
        # stack logtis
        logits_stacked = torch.zeros(
            batch_size,
            0,
            self.opt_model.config.vocab_size,
            device=outputs.logits[0].device,
        )
        for i in range(sequence_length):
            logits = outputs.logits[i].unsqueeze(1)
            logits_stacked = torch.cat([logits_stacked, logits], dim=1)

        outputs.logits = logits_stacked
        output_text = self.opt_tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        output_text = [text.strip() for text in output_text]
        outputs.predictions = output_text
        return outputs
