export TOKENIZERS_PARALLELISM=false;
gpus="'4'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=esol_mistral7b_string_only_10ep_1114 \
data=esol \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++trainer.max_epochs=10 \
++trainer.accumulate_grad_batches=11 \
++trainer.warmup_steps=1 \
++trainer.log_every_n_steps=1 \
++trainer.init_lr=1e-3 \
++trainer.min_lr=1e-4 \
++trainer.val_check_interval=1.0
