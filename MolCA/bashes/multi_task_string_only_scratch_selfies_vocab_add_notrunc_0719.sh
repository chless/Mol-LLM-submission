export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'
batch_size=288
inference_batch_size=320

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model 'facebook/galactica-1.3b' \
--tune_gnn \
--llm_tune lora \
--batch_size $batch_size \
--inference_batch_size $inference_batch_size \
--val_check_interval 5000 \
--max_epochs 20 \
--mol_representation string_only \
--num_beam 1 \
--gen_max_len=512 \
--prompt_max_len=512 \
--label_max_len=512 \
--save_every_n_epochs 1 \
--filename instruction_tuning_string_only_scratch_galac_selfies_vocab_add_notrunc_0719 \
--raw_data_root MolCA/data/multi_task_dataset_0715_selfies \
--apply_reg_order_scale \
--truncation 0 \
--padding longest