gpus=$1
python3 MolCA/stage2.py \
--task 'regression' \
--root 'qm9' \
--devices $gpus \
--filename "qm9_stage2_mse_backprop" \
--stage2_path 'MolCA/all_checkpoints/stage2_graph_embedding_mse/last.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 64 \
--val_check_interval 0.1 \
--max_epochs 1