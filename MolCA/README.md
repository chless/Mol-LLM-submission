# Introduction

The code is built upon [MolCA](https://github.com/acharkq/MolCA).
Thanks for the authors.

# Quick Start

For multi-task instruction tuning using 2D molecular graph and 1D SELFIES representation,
execute following command:

```
export TOKENIZERS_PARALLELISM=false;

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model facebook/galactica-1.3b \
--llm_tune lora \
--lora_r 64 \
--per_device_batch_size_cls 5 \
--per_device_batch_size_reg 10 \
--per_device_batch_size_rxn 10 \
--per_device_batch_size_rea 5 \
--per_device_batch_size_trn 5 \
--per_device_inference_batch_size_cls 70 \
--per_device_inference_batch_size_reg 70 \
--per_device_inference_batch_size_rxn 30 \
--per_device_inference_batch_size_rea 30 \
--per_device_inference_batch_size_trn 30 \
--val_check_interval 2500 \
--max_epochs 15 \
--second_stage_start_epoch 5 \
--num_beam 1 \
--raw_data_root MolCA/data/<YOUR_DATA_PATH> \
--gen_max_len 280 \
--prompt_max_len 280 \
--label_max_len 280 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--resize 2400 \
--skip_sanity_check \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string+graph \
--filename $filename\
```

# Quick Start

For multi-task instruction tuning using 2D molecular graph representation only,
execute following command:

```
export TOKENIZERS_PARALLELISM=false;

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model facebook/galactica-1.3b \
--llm_tune lora \
--lora_r 64 \
--per_device_batch_size_cls 5 \
--per_device_batch_size_reg 10 \
--per_device_batch_size_rxn 10 \
--per_device_batch_size_rea 5 \
--per_device_batch_size_trn 5 \
--per_device_inference_batch_size_cls 70 \
--per_device_inference_batch_size_reg 70 \
--per_device_inference_batch_size_rxn 30 \
--per_device_inference_batch_size_rea 30 \
--per_device_inference_batch_size_trn 30 \
--val_check_interval 2500 \
--max_epochs 15 \
--second_stage_start_epoch 5 \
--num_beam 1 \
--raw_data_root MolCA/data/<YOUR_DATA_PATH> \
--gen_max_len 280 \
--prompt_max_len 280 \
--label_max_len 280 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--resize 2400 \
--skip_sanity_check \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation graph_only \
--filename $filename\
```

# Quick Start

For multi-task instruction tuning using 2D molecular 1D SELFIES representation only,
execute following command:

```
export TOKENIZERS_PARALLELISM=false;

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model facebook/galactica-1.3b \
--llm_tune lora \
--lora_r 64 \
--per_device_batch_size_cls 5 \
--per_device_batch_size_reg 10 \
--per_device_batch_size_rxn 10 \
--per_device_batch_size_rea 5 \
--per_device_batch_size_trn 5 \
--per_device_inference_batch_size_cls 70 \
--per_device_inference_batch_size_reg 70 \
--per_device_inference_batch_size_rxn 30 \
--per_device_inference_batch_size_rea 30 \
--per_device_inference_batch_size_trn 30 \
--val_check_interval 2500 \
--max_epochs 15 \
--second_stage_start_epoch 5 \
--num_beam 1 \
--raw_data_root MolCA/data/<YOUR_DATA_PATH> \
--gen_max_len 280 \
--prompt_max_len 280 \
--label_max_len 280 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--resize 2400 \
--skip_sanity_check \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename $filename\
```