export TOKENIZERS_PARALLELISM=false;
gpus=$1

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode multi_task \
--filename instruction_tuning_string_only_galac_0628 \
--stage2_path 'MolCA/all_checkpoints/MolCA/stage2.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--inference_batch_size 840 \
--val_check_interval 0.1 \
--max_epochs 100 \
--mol_representation string_only \
--num_beam 1 \
--batch_size 840 \
--result_file MolCA/results/instruction_tuning_sting_only_galac_0628.csv \
--save_every_n_epochs 5