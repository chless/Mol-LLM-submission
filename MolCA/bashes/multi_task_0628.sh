export TOKENIZERS_PARALLELISM=false;
gpus=$1
batch_size=$2
inference_batch_size=$3

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
--batch_size $batch_size \
--inference_batch_size $inference_batch_size \
--val_check_interval 1 \
--max_epochs 10 \
--mol_representation string_only \
--num_beam 1 \

--save_every_n_epochs 1 \
--raw_data_root MolCA/data/multi_task_dataset_0630-2.pth