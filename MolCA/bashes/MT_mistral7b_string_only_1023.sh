export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"

python3 MolCA/stage3.py \
++devices=$gpus \
++filename=MT_mistral7b_string_only_1023 \
data=multi_task_extended \
trainer=mistral7b_80gb \
++trainer.mol_representation=string_only \
++ckpt_path=/home/chanhui-lee/text-mol/MolCA/all_checkpoints/MT_mistral7b_string_only_1023/step=01000-train_total_loss=0.000.ckpt
