gpus=$1

python3 MolCA/stage2.py \
--task 'regression2' \
--root 'qm9_homo' \
--devices $gpus \
--filename "qm9_homo_frozen_gnn2" \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 8 \
--val_check_interval 500 \
--max_epoch 2

python3 MolCA/stage2.py \
--task 'regression2' \
--root 'qm9_homo' \
--devices $gpus \
--filename "qm9_homo_frozen_gnn2" \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 8 \
--val_check_interval 500 \
--max_epoch 2

python3 MolCA/stage2.py \
--task 'regression2' \
--root 'qm9_homo' \
--devices $gpus \
--filename "qm9_homo_frozen_gnn2" \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 8 \
--val_check_interval 500 \
--max_epoch 2