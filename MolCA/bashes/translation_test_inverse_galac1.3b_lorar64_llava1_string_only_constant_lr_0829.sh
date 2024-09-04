export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3'

python3 MolCA/stage3.py \
--devices $gpus \
--root translation \
--mode test \
--llm_model facebook/galactica-1.3b \
--tune_llm lora \
--lora_r 64 \
--per_device_batch_size_trn 28 \
--per_device_inference_batch_size_trn 20 \
--val_check_interval 900 \
--max_epochs 100 \
--second_stage_start_epoch 5 \
--num_beam 5 \
--raw_data_root /data/datasets/multi_task_dataset_0829_selfies_transinverse \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir /data/text-mol-logging/tensorboard \
--graph_encoder_ckpt /data/ckpts/MoleculeSTM/molecule_model.pth \
--valset_resize 2400 \
--skip_sanity_check \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename translation_galac1.3b_lorar64_llava1_string_only_constant_lr_0826 \
--scheduler None \
--stage2_path /data/ckpts/molllm-ckpt/MT_galac1.3b_lorar64_llava1_string_only_constant_lr_0822-2/step=130000-total_loss=0.374.ckpt