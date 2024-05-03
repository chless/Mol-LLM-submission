gpus=$1
filename=$2
python MolCA/stage1.py \
--root 'MolCA/data/PubChem324kV2/' \
--gtm \
--lm \
--devices $gpus \
--mode test \
--filename $filename \
--rerank_cand_num 128 \
--num_query_token 8 \
--tune_gnn
