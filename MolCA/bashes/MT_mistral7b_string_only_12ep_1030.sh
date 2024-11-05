export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_string_only_12ep_1030 \
data=multi_task_extended-v2 \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++ckpt_path="'/home/chanhui-lee/text-mol/MolCA/all_checkpoints/MT_mistral7b_string_only_12ep_1030/last-v2.ckpt'"
