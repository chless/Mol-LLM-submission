
python3 MolCA/stage2.py \
--task 'regression' \
--root 'qm9' \
--devices '4,5' \
--filename "qm9_stage2_gnn" \
--stage2_path 'MolCA/all_checkpoints/MolCA/stage2.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--max_epochs 1 \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 8 \
--val_check_interval 0.1 \
--max_epoch 1