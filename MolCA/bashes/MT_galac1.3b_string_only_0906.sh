export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--llm_model facebook/galactica-1.3b \
--tune_llm lora \
--lora_r 64 \
--per_device_batch_size_cls 6 \
--per_device_batch_size_reg 6 \
--per_device_batch_size_rxn 6 \
--per_device_batch_size_rea 6 \
--per_device_batch_size_trn 6 \
--per_device_inference_batch_size_cls 80 \
--per_device_inference_batch_size_reg 80 \
--per_device_inference_batch_size_rxn 40 \
--per_device_inference_batch_size_rea 40 \
--per_device_inference_batch_size_trn 40 \
--val_check_interval 2500 \
--max_epochs 100 \
--second_stage_start_step 25000 \
--num_beam 1 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--valset_resize 2400 \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename MT_galac1.3b_string_only_0906 \
--scheduler None \