"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.nn as nn
from torch.amp import autocast as autocast
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

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import replace_return_docstrings

from typing import Optional, List, Tuple, Union


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
        tune_llm="freeze",
        peft_dir="",
        llm_model="facebook/galactica-1.3b",
        prompt="",  # TODO: remove. currently LLM classes not use prompt from args.prompt
        args=None,
    ):
        super().__init__()
        self.args = args
        self.peft_dir = peft_dir

        # initialize opt model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, use_fast=False, padding_side="right"
        )
        self.llm_tokenizer.mol_string_randomization_ratio = (
            args.mol_string_randomization_ratio
        )
        self.add_necessary_tokens()

        self.set_llm_model(llm_model)

        self.llm_model.resize_token_embeddings(
            len(self.llm_tokenizer)
        )  # this will cause bug when full fine-tuning the opt model

        self.tune_llm = tune_llm
        if tune_llm == "lora":
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(
                    self.llm_model, peft_dir, is_trainable=True
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
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        elif tune_llm == "freeze":
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif tune_llm == "full":
            pass
        else:
            raise NotImplementedError()

        self.set_params_requires_grads(
            model=self.llm_model, keyword="embed", grad=True, IsPrint=False
        )

        if self.args.llava_style:
            self.set_params_requires_grads(
                model=self.llm_model, keyword="lora", grad=False, IsPrint=False
            )

        if "graph" in self.args.mol_representation:
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

            self.opt_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
            )

    def fit_llm_input_convention(self, llm_prompt):
        llm_prompt = (
            added_tokens.INSTRUCTION[0] + llm_prompt + added_tokens.INSTRUCTION[1]
        )
        return llm_prompt

    def fit_llm_output_convention(self, llm_output):
        # TODO: if train galactica next time, use the commented code. The code is commented currently because previously trained model is not aligned with.
        llm_output += self.llm_tokenizer.eos_token
        return llm_output

    def set_llm_model(self, llm_model):
        if llm_model == "facebook/galactica-125m":
            self.llm_model = OPTForCausalLM_Custom.from_pretrained(
                llm_model, torch_dtype=torch.bfloat16
            )
        else:
            if torch.cuda.is_bf16_supported():
                self.llm_model = OPTForCausalLM_Custom.from_pretrained(
                    llm_model, torch_dtype=torch.bfloat16
                )
            else:
                self.llm_model = OPTForCausalLM_Custom.from_pretrained(
                    llm_model, torch_dtype=torch.float16
                )

    def add_special_token(self):
        # pad toekn for galactica is "<pad>""
        self.llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "\n"})

    def add_necessary_tokens(self):
        self.add_special_token()

        if self.args.add_selfies_tokens:
            # Read txt from selfies_token_path
            with open(self.args.selfies_token_path, "r") as f:
                selfies_tokens = f.readlines()
                selfies_tokens = [token.strip() for token in selfies_tokens]
            self.llm_tokenizer.add_tokens(selfies_tokens)
            # get token id of the selfies_tokens
            self.llm_tokenizer.selfies_token_ids = [
                self.llm_tokenizer(token, add_special_tokens=False).input_ids[0]
                for token in selfies_tokens
            ]
            self.llm_tokenizer.added_selfies_tokens = selfies_tokens
            # remove '.' from the marked list for selfies token
            # self.llm_tokenizer.added_selfies_tokens.remove(".")
            # self.llm_tokenizer.selfies_token_ids.remove(36)
            print(f"Added {len(selfies_tokens)} selfies tokens to the tokenizer")

        additional_tokens = [
            getattr(added_tokens, tokens)
            for tokens in dir(added_tokens)
            if not re.match("__.*__", tokens)
        ]
        additional_tokens = [
            token for sublist in additional_tokens for token in sublist
        ]

        self.llm_tokenizer.add_tokens(additional_tokens)

        self.llm_tokenizer.mol_token = added_tokens.MOL_EMBEDDING[0]
        self.llm_tokenizer.mol_ph_token = self.llm_tokenizer.mol_token * self.args.num_query_token
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer(
            self.llm_tokenizer.mol_token, add_special_tokens=False
        ).input_ids[0]

    def merge_and_initialize_lora(self):
        self.model.blip2model.llm_model.merge_and_unload(progressbar=True)

        if self.tune_llm == "lora":
            if self.peft_dir:
                self.llm_model = PeftModel.from_pretrained(
                    self.llm_model, self.peft_dir, is_trainable=True
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
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        elif self.tune_llm == "freeze":
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif self.tune_llm == "full":
            pass
        else:
            raise NotImplementedError()

    def random_replace_mol_string(self, input_tokens_input_ids):
        ids = input_tokens_input_ids
        tokenizer = self.llm_tokenizer
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

    def forward(self, batch):
        graphs, input_tokens, target_tokens = batch
        
        # TODO: currently not using, but not determined to derprecate or not
        if self.args.mol_string_randomization_ratio > 0:
            input_tokens.input_ids = self.random_replace_mol_string(
                input_tokens.input_ids
            )

        # preprare targets to ignore pad tokens in the loss calculation
        targets = target_tokens.input_ids.masked_fill(
            target_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )

        input_embeds = self.llm_model.get_input_embeddings()(input_tokens.input_ids)
        if "graph" in self.args.mol_representation:
            input_embeds = self.inject_graph_embeds2input_embeds(
                input_embeds=input_embeds,
                input_tokens=input_tokens,
                graphs=graphs,
            )

        outputs = self.llm_model(
            inputs_embeds=input_embeds,
            attention_mask=input_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        """
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
        """
        results = {
            "loss": outputs.loss,
            "instance_loss": outputs.instance_loss,
            "logits": outputs.logits,
        }
        return results

    def inject_graph_embeds2input_embeds(self, input_embeds, input_tokens, graphs):
        tasks = graphs.integrated_seq['task_subtask_pairs']
        double_mol_idxs = [True if "reagent_prediction" in task else False for task in tasks]
        if "additional_x" in graphs.keys():
            mol_token_sequence = []
            for prefix in ["", "additional_"]:
                mol_x = graphs[f"{prefix}x"]
                mol_edge_index = graphs[f"{prefix}edge_index"]
                mol_edge_attr = graphs[f"{prefix}edge_attr"]
                mol_batch = graphs[f"{prefix}batch"]
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

        # [Batch_size, Sequence_length, Hidden_size]
        # data_idx over Batch_size, query_idx over Sequence_length
        for data_idx in range(input_tokens.is_mol_token.shape[0]):
            # only inject mol tokens to the prompt embeds when there is mol token in the prompt
            mol_token_indices = input_tokens.is_mol_token[data_idx]
            num_mol_tokens_in_prompt = mol_token_indices.sum().item()
            if num_mol_tokens_in_prompt:
                # TODO: fix the bug that shapes are not matched.
                input_embeds[data_idx, mol_token_indices, :] = mol_tokens[data_idx, :num_mol_tokens_in_prompt]
            else:
                pass
        return input_embeds

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
        input_tokens = samples["input_tokens"]

        input_embeds = self.llm_model.get_input_embeddings()(input_tokens.input_ids)
        if "graph" in self.args.mol_representation:
            input_embeds = self.inject_graph_embeds2input_embeds(
                input_embeds=input_embeds,
                input_tokens=input_tokens,
                graphs=graphs,
            )

        outputs = self.llm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            # min_length=min_length,
            min_new_tokens=min_length,  # TODO: change to min_new_tokens for all layered methods
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
        )

        batch_size, sequence_length = outputs.sequences.shape
        # stack logtis
        logits_stacked = torch.zeros(
            batch_size,
            0,
            self.llm_model.config.vocab_size,
            device=outputs.logits[0].device,
        )
        for i in range(sequence_length):
            logits = outputs.logits[i].unsqueeze(1)
            logits = (
                logits.view(batch_size, num_beams, -1).max(dim=1).values.unsqueeze(1)
            )
            logits_stacked = torch.cat([logits_stacked, logits], dim=1)

        outputs.logits = logits_stacked
        output_text = self.llm_tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )

        output_text = [text.strip() for text in output_text]
        outputs.predictions = output_text
        return outputs


_CONFIG_FOR_DOC = "OPTConfig"


class OPTForCausalLM_Custom(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM_Custom.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            # custom forward to get not reduced loss
            loss_fct_not_reduced = CrossEntropyLoss(reduction="none")
            loss_not_reduced = loss_fct_not_reduced(shift_logits, shift_labels).view(
                labels.size(0), -1
            )
            # normalization exclude default ignore index -100
            instance_non_pad_tokens = torch.where(
                shift_labels != -100,
                torch.tensor(1).to(shift_labels.device),
                torch.tensor(0).to(shift_labels.device),
            ).view(labels.size(0), -1)
            instance_loss = (loss_not_reduced * instance_non_pad_tokens).sum(
                dim=-1
            ) / instance_non_pad_tokens.sum(dim=-1)
            instance_loss = instance_loss.detach()
            # cross entropy aggregate not row-wise, but sum of all instances
            loss = (
                loss_not_reduced * instance_non_pad_tokens
            ).sum() / instance_non_pad_tokens.sum()
        else:
            instance_loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast_Custom(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            instance_loss=instance_loss,
        )


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class CausalLMOutputWithPast_Custom(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    instance_loss: Optional[torch.FloatTensor] = None
