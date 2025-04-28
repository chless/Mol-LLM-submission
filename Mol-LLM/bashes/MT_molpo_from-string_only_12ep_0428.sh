export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
gnn=gine_custom
file_name=MT_molpo_from-string_only_${gnn}_12ep_0428
molpo_lambda=$1
rejected_lambda=$2
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
trainer.molpo_lambda=$molpo_lambda \
trainer.rejected_lambda=$rejected_lambda \
pretrained_ckpt_path="'/data/all_checkpoints/MT_mistral7b_string+graph_qformer_${gnn}_pretraining_1ep_0415/epoch=00-step=3415.ckpt'"

