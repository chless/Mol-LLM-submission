export TOKENIZERS_PARALLELISM=false;
gpus="'4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=qm9_homo_galac1.3b_string_only_0927 \
data=qm9_homo \
trainer=vram80gb_lr1e-4 \
++trainer.accumulate_grad_batches=1 \
++data.val_check_interval=600