gpus=$1

python3 MolCA/stage2.py \
--task 'regression' \
--root 'qm9_homo' \
--devices $gpus \
--filename "ft-qm9_homo_qformer_instruction_0417" \
--opt_model 'facebook/galactica-1.3b' \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--inference_batch_size 8 \
--val_check_interval 300 \
--max_epoch 1 \
--qformer_instruction