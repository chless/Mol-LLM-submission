export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model facebook/galactica-1.3b \
--tune_gnn \
--llm_tune lora \
--lora_r 64 \
--per_device_batch_size_cls 3 \
--per_device_batch_size_reg 18 \
--per_device_batch_size_rxn 12 \
--per_device_batch_size_rea 6 \
--per_device_batch_size_trn 6 \
--per_device_inference_batch_size_cls 3 \
--per_device_inference_batch_size_reg 18 \
--per_device_inference_batch_size_rxn 12 \
--per_device_inference_batch_size_rea 6 \
--per_device_inference_batch_size_trn 6 \
--val_check_interval 2500 \
--max_epochs 10 \
--mol_representation string_only \
--num_beam 1 \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir MolCA/all_checkpoints \
--filename MT_galac1.3b_lorar64_llava1_0813 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--resize 2400 \
--skip_sanity_check \
--llava_style 1