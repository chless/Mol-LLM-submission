export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_string_only_10ep_1119 \
data=multi_task \
trainer=mistral7b_lorar32_80gb \
++trainer.mol_representation=string_only \
++ckpt_path="'MolCA/all_checkpoints/MT_mistral7b_string_only_12ep_1119/last.ckpt'"
