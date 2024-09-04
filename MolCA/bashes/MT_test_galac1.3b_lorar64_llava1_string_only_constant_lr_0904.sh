export TOKENIZERS_PARALLELISM=false;
gpus='3,4,5'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode test \
--llm_model facebook/galactica-1.3b \
--tune_llm lora \
--lora_r 64 \
--per_device_inference_batch_size_trn 30 \
--per_device_inference_batch_size_rea 30 \
--per_device_inference_batch_size_rxn 30 \
--per_device_inference_batch_size_reg 60 \
--per_device_inference_batch_size_cls 60 \
--num_beam 1 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt /data/ckpts/MoleculeSTM/molecule_model.pth \
--skip_sanity_check \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename MT_galac1.3b_lorar64_llava1_string_only_constant_lr_0904 \
--stage2_path MolCA/all_checkpoints/MT_galac1.3b_lorar64_llava1_string_only_constant_lr_0822/step=130000-total_loss=0.374.ckpt \
--scheduler None \
--truncation 1 \
--padding max_length \
--test_on_trainset \
--trainset_resize 4800