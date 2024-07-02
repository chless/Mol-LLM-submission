export TOKENIZERS_PARALLELISM=false;
gpus=$1
batch_size=$2
inference_batch_size=$3

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode ft \
--filename instruction_tuning_string+graph_ours_llama3_8b_0703 \
--stage2_path '/home/chanhui-lee/text-mol/MolCA/all_checkpoints/ft-llava-tunegnn-authorcode/epoch=99.ckpt' \
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
--result_file MolCA/results/instruction_tuning_sting+graph_ours_llama3_8b_0703.csv \
--save_every_n_epochs 1 \
--raw_data_root MolCA/data/multi_task_dataset