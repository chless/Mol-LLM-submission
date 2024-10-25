export TOKENIZERS_PARALLELISM=false;
gpus="'1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_string_only_left_1025 \
data=multi_task_extended \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++trainer.accumulate_grad_batches=14
