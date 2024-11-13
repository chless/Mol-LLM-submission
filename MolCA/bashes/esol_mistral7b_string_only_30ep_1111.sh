export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=esol_mistral7b_string_only_30ep_1111 \
data=esol \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++trainer.max_epochs=30 \
++trainer.accumulate_grad_batches=4
