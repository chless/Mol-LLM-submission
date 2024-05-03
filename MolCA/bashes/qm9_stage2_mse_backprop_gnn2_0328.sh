gpus=$1
python3 MolCA/stage2.py \
--task 'regression2' \
--root 'qm9_homo' \
--devices $gpus \
--filename "qm9_homo_stage2_mse_backprop2" \
--stage2_path 'MolCA/all_checkpoints/stage2_graph_embedding_mse/last.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 64 \
--val_check_interval 500 \
--max_epochs 2

python3 MolCA/stage2.py \
--task 'regression2' \
--root 'qm9_lumo' \
--devices $gpus \
--filename "qm9_lumo_stage2_mse_backprop2" \
--stage2_path 'MolCA/all_checkpoints/stage2_graph_embedding_mse/last.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 64 \
--val_check_interval 500 \
--max_epochs 2

python3 MolCA/stage2.py \
--task 'regression2' \
--root 'qm9_homo_lumo_gap' \
--devices $gpus \
--filename "qm9_homo_lumo_gap_stage2_mse_backprop2" \
--stage2_path 'MolCA/all_checkpoints/stage2_graph_embedding_mse/last.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 64 \
--val_check_interval 500 \
--max_epochs 2