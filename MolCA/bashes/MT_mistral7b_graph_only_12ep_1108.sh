export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_graph_only_12ep_1108 \
data=multi_task_extended-v5 \
trainer=mistral7b_80gb \
++trainer.mol_representation=graph_only \
