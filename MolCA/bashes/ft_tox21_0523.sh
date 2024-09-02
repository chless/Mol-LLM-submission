gpus=$1
filename=$2

python3 MolCA/stage3.py \
--root 'tox21' \
--devices $gpus \
--filename $filename \
--stage2_path 'MolCA/all_checkpoints/MolCA/stage2.ckpt' \
--llm_model 'facebook/galactica-1.3b' \
--max_epochs 20 \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--tune_llm lora \
--inference_batch_size 1 \
--val_check_interval 1 \
--max_epochs 10 \
--batch_size 64 \
--num_beam 1