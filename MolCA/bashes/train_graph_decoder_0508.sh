gpus=$1
filename=$2
python MolCA/train_graph_decoder.py \
--root 'MolCA/data/PubChem324kV2/' \
--devices $gpus \
--mode train \
--filename $filename \
--tune_gnn \
--max_epoch 20 \
--save_every_n_epochs 1 \
