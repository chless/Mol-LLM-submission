gpus=$1
python MolCA/stage1.py \
--root 'MolCA/data/PubChem324kV2/' \
--gtm \
--lm \
--devices $gpus \
--mode train \
--filename stage1_query16_0418 \
--rerank_cand_num 128 \
--num_query_token 16 \
--tune_gnn \
--filtered_cid_path data/PubChem324k/filtered_pretrain_cids.txt
