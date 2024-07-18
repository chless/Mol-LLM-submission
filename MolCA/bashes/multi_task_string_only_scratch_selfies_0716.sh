export TOKENIZERS_PARALLELISM=false;
gpus=$1
batch_size=$2
inference_batch_size=$3

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--opt_model 'facebook/galactica-1.3b' \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--batch_size $batch_size \
--inference_batch_size $inference_batch_size \
--val_check_interval 5000 \
--max_epochs 20 \
--mol_representation string_only \
--num_beam 1 \
--max_len=512 \
--text_max_len=256 \
--save_every_n_epochs 1 \
--stage2_path '/home/chanhui-lee/text-mol/MolCA/all_checkpoints/instruction_tuning_string_only_scratch_selfies_galac_0716/last-v3.ckpt' \
--filename instruction_tuning_string_only_scratch_selfies_galac_0716 \

--raw_data_root MolCA/data/multi_task_dataset_0715_selfies \