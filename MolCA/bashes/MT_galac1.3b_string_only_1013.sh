export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_galac1.3b_string_only_1013 \
data=multi_task_extended \
trainer=8b_80gb \
++trainer.mol_representation=string_only \
llm=galactica1.3b
