export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=gine_custom
file_name=MT_string+graph_12+6ep_0430
max_epochs=6
gnn=gine_custom

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=multi_task \
gnn=${gnn} \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string_only_12ep_0415/epoch=11-step=40991.ckpt'" \
trainer.init_lr=0.00004 \
trainer.warmup_lr=0.000004 \
trainer.val_check_interval=0.2

