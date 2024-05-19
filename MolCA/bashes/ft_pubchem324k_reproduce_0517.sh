gpus=7
filename=$1
batch_size=64

python3 MolCA/stage2.py \
--root 'MolCA/data/PubChem324kV2/' \
--devices $gpus \
--filename $filename \
--stage2_path '/home/chanhui-lee/text-mol/MolCA/all_checkpoints/MolCA/stage2.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--max_epochs 100 \
--mode ft \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--inference_batch_size 8 \
--batch_size $batch_size