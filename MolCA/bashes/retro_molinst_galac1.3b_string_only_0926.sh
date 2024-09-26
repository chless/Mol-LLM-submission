export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=retrosynthesis_molinst_galac1.3b_string_only_0926 \
data=retrosynthesis_molinst \
trainer=retro_string_only \
