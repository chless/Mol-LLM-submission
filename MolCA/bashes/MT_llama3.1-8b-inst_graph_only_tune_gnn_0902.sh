export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--llm_model meta-llama/Meta-Llama-3.1-8B-Instruct \
--tune_llm lora \
--tune_gnn \
--lora_r 64 \
--per_device_batch_size_cls 2 \
--per_device_batch_size_reg 4 \
--per_device_batch_size_rxn 4 \
--per_device_batch_size_rea 2 \
--per_device_batch_size_trn 2 \
--per_device_inference_batch_size_trn 6 \
--per_device_inference_batch_size_reg 12 \
--per_device_inference_batch_size_rea 6 \
--per_device_inference_batch_size_rxn 6 \
--per_device_inference_batch_size_cls 12 \
--val_check_interval 8000 \
--max_epochs 100 \
--second_stage_start_step 10000 \
--num_beam 1 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--resize 2400 \
--llava_style 1 \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation graph_only \
--filename MT_llama3.1-8b-inst_graph_only_tune_gnn_0902 \
--scheduler None \