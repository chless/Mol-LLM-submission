export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3'
inference_batch_size=128

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode test \
--opt_model 'facebook/galactica-1.3b' \
--tune_gnn \
--llm_tune lora \
--inference_batch_size $inference_batch_size \
--mol_representation string_only \
--num_beam 1 \
--max_len=512 \
--text_max_len=512 \
--save_every_n_epochs 1 \
--filename instruction_tuning_string_only_scratch_galac_selfies_vocab_add_0717 \
--stage2_path MolCA/all_checkpoints/instruction_tuning_string_only_scratch_galac_selfies_vocab_add_0717/step=99000-total_loss=1.390.ckpt \
--raw_data_root MolCA/data/multi_task_dataset_0715_selfies