gpus=$1
filename=$2

python3 MolCA/stage2.py \
--root 'MolCA/data/PubChem324kV2/' \
--devices $gpus \
--filename $filename \
--stage2_path '/home/chanhui-lee/text-mol/MolCA/all_checkpoints/ft_pubchem324k_reproduce_0508/epoch=99.ckpt' \
--peft_dir '/home/chanhui-lee/text-mol/MolCA/all_checkpoints/ft_pubchem324k_reproduce_0508/epoch=99.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--max_epochs 100 \
--mode eval \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--inference_batch_size 8 \