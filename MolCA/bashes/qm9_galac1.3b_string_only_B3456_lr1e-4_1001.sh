export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=qm9_galac1.3b_string_only_B3456_lr1e-4_1001 \
data=qm9 \
trainer=vram80gb_lr1e-4 \
++trainer.accumulate_grad_batches=9 \
++trainer.val_check_interval=900 \
++ckpt_path="'MolCA/all_checkpoints/qm9_galac1.3b_string_only_B388_lr1e-4_0929/step=44200-train_total_loss=0.311.ckpt'"