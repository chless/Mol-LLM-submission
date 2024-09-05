export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode test \
--llm_model meta-llama/Meta-Llama-3.1-8B-Instruct \
--tune_llm lora \
--lora_r 64 \
--per_device_inference_batch_size_trn 4 \
--per_device_inference_batch_size_rea 4 \
--per_device_inference_batch_size_rxn 4 \
--per_device_inference_batch_size_reg 8 \
--per_device_inference_batch_size_cls 8 \
--num_beam 1 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--gen_max_len 222 \
--prompt_max_len 222 \
--label_max_len 222 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--valset_resize 2400 \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename MT_test_trainset_llama3.1-8b-inst_string_only_test_0904 \
--stage2_path MolCA/all_checkpoints/MT_llama3.1-8b-inst_string_only_0903/step=89000-train/total_loss=0.384.ckpt \
--scheduler None \
--truncation 1 \
--padding max_length \