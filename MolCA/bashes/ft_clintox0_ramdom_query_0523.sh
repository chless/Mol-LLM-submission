gpus=$1
task=clintox

rqs=(8 6 3 0)
for rq in ${rqs[@]}; do
    echo ${task}:random_query${rq}
    python3 MolCA/stage3.py \
    --root $task \
    --subtask_idx 0 \
    --devices $gpus \
    --filename ft_${task}0_rq${rq}_0523 \
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
done