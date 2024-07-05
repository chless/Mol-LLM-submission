export TOKENIZERS_PARALLELISM=false;
gpus='4,5,6,7'
batch_size=32
inference_batch_size=3040

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode test \
--filename instruction_tuning_string_only_galac_0628 \
--stage2_path 'MolCA/all_checkpoints/instruction_tuning_string_only_galac_0628/last-v2.ckpt' \
--result_file MolCA/all_checkpoints/instruction_tuning_string_only_galac_0628/benchmark_performance.json \
--opt_model 'facebook/galactica-1.3b' \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--batch_size $batch_size \
--inference_batch_size $inference_batch_size \
--val_check_interval 0.1 \
--max_epochs 10 \
--mol_representation string_only \
--num_beam 1 \
--save_every_n_epochs 1 \
--raw_data_root MolCA/data/multi_task_dataset