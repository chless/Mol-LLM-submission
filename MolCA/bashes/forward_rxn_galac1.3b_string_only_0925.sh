export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=forward_rxn_galac1.3b_string_only_0925 \
data=forward_reaction_prediction \
trainer=forward_string_only \
