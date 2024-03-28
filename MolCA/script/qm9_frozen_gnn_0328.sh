gpus=$1

python3 MolCA/stage2.py \
--task 'regression' \
--root 'qm9' \
--devices $gpus \
--filename "qm9_frozen_gnn" \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 8 \
--val_check_interval 0.1 \
--max_epoch 1