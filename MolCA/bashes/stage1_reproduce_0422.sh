gpus=6
filename=$1
batch_size=32
python MolCA/stage1.py \
--root 'MolCA/data/PubChem324kV2/' \
--gtm \
--lm \
--devices $gpus \
--mode train \
--filename $filename \
--rerank_cand_num 128 \
--num_query_token 8 \
--tune_gnn \
--batch_size $batch_size
