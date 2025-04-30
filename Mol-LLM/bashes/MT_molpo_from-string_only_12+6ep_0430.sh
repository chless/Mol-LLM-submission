export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=MT_molpo_from-string_only_12+6ep_0430

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=molpo \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_from-string_only_qformer_pretraining_1ep_0430/last.ckpt'"

