export TOKENIZERS_PARALLELISM=false;
gpus="'4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=qm9_galac1.3b_string_only_B388_lr1e-4_0929 \
data=qm9 \
trainer=vram80gb_lr1e-4 \
++trainer.accumulate_grad_batches=6 \
++trainer.val_check_interval=1800 \