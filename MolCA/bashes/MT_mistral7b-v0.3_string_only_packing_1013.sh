export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b-v0.3_string_only_packing_1013 \
data=multi_task_extended \
trainer=8b_80gb_packing \
++trainer.mol_representation=string_only \
llm=mistral-7b-instruct-v0.3
