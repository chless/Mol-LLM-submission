export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_galac1.3b_string_only_1026 \
data=multi_task_extended-v2 \
trainer=galactica1.3b_80gb \
++trainer.mol_representation=string_only \