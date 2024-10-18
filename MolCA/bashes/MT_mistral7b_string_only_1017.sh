export TOKENIZERS_PARALLELISM=false;
gpus="'5'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_string_only_1017 \
data=multi_task_extended \
trainer=mistral_8b_80gb \
++trainer.mol_representation=string_only \
llm=mistral-7b-instruct-v0.3
