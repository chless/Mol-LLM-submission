task=$1
subtask=$2
gpus=$3
rq=$4
echo $rq

python3 MolCA/stage3.py \
--root $task \
--subtask_idx $subtask \
--devices $gpus \
--filename ft_${task}${subtask}_rq${rq}_0523 \
--stage2_path 'MolCA/all_checkpoints/MolCA/stage2.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--inference_batch_size 8 \
--val_check_interval 0.1 \
--max_epochs 10 \
--batch_size 64 \
--num_beam 1 \
--num_random_query_embedding $rq