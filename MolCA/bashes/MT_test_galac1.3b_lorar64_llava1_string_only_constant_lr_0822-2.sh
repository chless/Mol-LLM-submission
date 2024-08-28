export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3'

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
--per_device_inference_batch_size_cls 14 \
--per_device_inference_batch_size_reg 14 \
--per_device_inference_batch_size_rxn 8 \
--per_device_inference_batch_size_rea 8 \
--per_device_inference_batch_size_trn 8 \
--val_check_interval 2500 \
--max_epochs 100 \
--second_stage_start_epoch 1 \
--num_beam 5 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--resize 2400 \
--skip_sanity_check \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename MT_galac1.3b_lorar64_llava1_string_only_constant_lr_0822-2 \
--stage2_path /data/ckpts/molllm-ckpt/MT_galac1.3b_lorar64_llava1_string_only_constant_lr_0822-2/step=130000-total_loss=0.374.ckpt \
--scheduler None \