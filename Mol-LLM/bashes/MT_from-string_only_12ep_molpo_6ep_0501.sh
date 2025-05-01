export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=$1
file_name=MT_molpo-${gnn}_from-string_only_12+6ep_0501

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data=molpo \
gnn=$gnn \
trainer=mistral7b_80gb_molpo \
trainer.mol_representation=string+graph \
trainer.skip_sanity_check=false \
pretrained_ckpt_path="'/data/all_checkpoints/MT_from-string_only_qformer-${gnn}_pretraining_1ep_0430/last.ckpt'"

