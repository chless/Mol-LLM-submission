export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode test \
--llm_model facebook/galactica-1.3b \
--tune_llm lora \
--lora_r 64 \
--per_device_inference_batch_size_trn 70 \
--per_device_inference_batch_size_rea 70 \
--per_device_inference_batch_size_rxn 70 \
--per_device_inference_batch_size_reg 120 \
--per_device_inference_batch_size_cls 120 \
--num_beam 1 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--gen_max_len 256 \
--prompt_max_len 256 \
--label_max_len 256 \
--logging_dir MolCA/all_checkpoints \
--graph_encoder_ckpt MolCA/MoleculeSTM/molecule_model.pth \
--num_query_token 32 \
--bert_num_hidden_layers 5 \
--mol_string_randomization_ratio -1 \
--mol_representation string_only \
--filename MT_galac1.3b_string_only_test_0909 \
--scheduler None \
--truncation 1 \
--padding max_length \
--test_on_trainset \
--stage2_path MolCA/all_checkpoints/MT_galac1.3b_string_only_grad_accum8_0911/last.ckpt