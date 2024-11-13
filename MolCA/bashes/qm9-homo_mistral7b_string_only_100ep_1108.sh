export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=qm9-homo_mistral7b_string_only_100ep_1108 \
data=qm9_homo \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++trainer.max_epochs=120 \
++trainer.accumulate_grad_batches=1
