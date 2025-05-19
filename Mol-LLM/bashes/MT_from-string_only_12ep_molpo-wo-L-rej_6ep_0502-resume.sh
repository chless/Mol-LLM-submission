export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=MT_from-string_only_12ep_molpo-wo-L-rej_6ep_0502

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=molpo \
trainer=mistral7b_80gb_molpo \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=true \
ckpt_path="'/data/all_checkpoints/MT_from-string_only_12ep_molpo-wo-L-rej_6ep_0502/epoch=02-step=3875.ckpt'" \
trainer.anc_rejected_weight=0.0 \
wandb_id=k6eale8n


