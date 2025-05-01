export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
file_name=MT_from-string_only_12ep_molpo_6ep_0502

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=molpo \
trainer=mistral7b_80gb_molpo \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_from-string_only_qformer-gine_tokengt_pretraining_1ep_0501/epoch=00-step=3415.ckpt'"

