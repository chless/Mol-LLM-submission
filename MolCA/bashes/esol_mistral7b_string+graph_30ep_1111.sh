export TOKENIZERS_PARALLELISM=false;
gpus="'7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=esol_mistral7b_string+graph_30ep_1111 \
data=esol \
trainer=mistral7b_80gb \
++trainer.mol_representation=string+graph \
++trainer.max_epochs=30 \
++trainer.accumulate_grad_batches=8 \
++trainer.warmup_steps=1 \
++trainer.log_every_n_steps=1
