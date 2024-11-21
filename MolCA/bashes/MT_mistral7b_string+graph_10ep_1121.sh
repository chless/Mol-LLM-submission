export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_string+graph_10ep_1121 \
data=multi_task_extended-v6 \
trainer=mistral7b_lorar32_80gb \
++trainer.mol_representation=string+graph
++trainer.
