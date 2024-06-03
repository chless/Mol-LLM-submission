gpus=$1

python3 MolCA/stage2.py \
--root 'MolCA/data/PubChem324kV2/' \
--devices $gpus \
--filename "stage2_graph_embedding_mse_logging" \
--opt_model 'facebook/galactica-1.3b' \
--max_epochs 10 \
--mode pretrain \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune freeze \
--inference_batch_size 4 \
--caption_eval_epoch 10 \
--filtered_cid_path MolCA/data/PubChem324k/filtered_pretrain_cids.txt 