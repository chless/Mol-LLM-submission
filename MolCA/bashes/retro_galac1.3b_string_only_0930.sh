export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=retrosynthesis+smol_galac1.3b_string_only_0930 \
data=retrosynthesis+smol \
trainer=vram80gb \
++ckpt_path="'MolCA/all_checkpoints/retrosynthesis_galac1.3b_string_only_0925/last-v1.ckpt'"
