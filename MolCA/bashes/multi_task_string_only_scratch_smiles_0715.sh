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
--val_check_interval 3000 \
--max_epochs 10 \
--mol_representation string_only \
--num_beam 1 \
--max_len=512 \
--text_max_len=256 \
--save_every_n_epochs 1 \
--filename instruction_tuning_string_only_scratch_smiles_galac_0715 \

--raw_data_root MolCA/data/multi_task_dataset_0715_smiles \