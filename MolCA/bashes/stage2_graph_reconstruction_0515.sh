gpus=$1
filename=$2

python3 MolCA/stage2.py \
--root 'MolCA/data/PubChem324kV2/' \
--devices $gpus \
--filename $filename \
--stage1_path '/home/chanhui-lee/text-mol/MolCA/all_checkpoints/MolCA/stage1.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--max_epochs 10 \
--mode pretrain \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune freeze \
--inference_batch_size 4 \
--graph_reconstruction