export TOKENIZERS_PARALLELISM=false;
gpus='3'
batch_size=36
inference_batch_size=36

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--llm_model 'facebook/galactica-1.3b' \
--tune_gnn \
--llm_tune lora \
--batch_size $batch_size \
--inference_batch_size $inference_batch_size \
--val_check_interval 5 \
--max_epochs 20 \
--mol_representation string_only \
--num_beam 1 \
--gen_max_len=256 \
--prompt_max_len=256 \
--label_max_len=256 \
--save_every_n_epochs 1 \
--filename instruction_tuning_string_only_scratch_galac_selfies_vocab_add_0730 \
--raw_data_root MolCA/data/multi_task_dataset_0813_selfies \
--resize 100 \
--skip_sanity_check