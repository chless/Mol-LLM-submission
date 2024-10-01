export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=reaction+smol_galac1.3b_string_only_1001 \
data=reaction+smol \
trainer=vram80gb \
++trainer.accumulate_grad_batches=16
