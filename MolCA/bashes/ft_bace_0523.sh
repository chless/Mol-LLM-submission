gpus=$1
filename=$2

python3 MolCA/stage3.py \
--root 'bace' \
--devices $gpus \
--filename $filename \
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
--num_beam 1 