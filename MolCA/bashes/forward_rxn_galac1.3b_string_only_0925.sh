export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3,4,5,6,7'

python3 MolCA/stage3.py \
++devices $gpus \
data=forward_reaction_prediction \
trainer=forward_string_only
