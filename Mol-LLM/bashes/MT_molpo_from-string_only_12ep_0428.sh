export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=gine_custom
file_name=MT_molpo_from-string_only_${gnn}_12ep_0428
max_epochs=3

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=molpo \
gnn=${gnn} \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string_only_12ep_0415/epoch=09-step=32451.ckpt'" \
trainer.init_lr=0.00004 \
trainer.warmup_lr=0.000004

