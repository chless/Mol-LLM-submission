export TOKENIZERS_PARALLELISM=false;
gpus=$1
data_tag=$2
filename=test_${data_tag}_mol_llm_wo_molpo_0514
ckpt_path="'/data/all_checkpoints/MT_from-string_only_12ep_string+graph_6ep_0502_continue/epoch=00-step=3229.ckpt'"

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
data.data_tag=${data_tag} \
trainer=mistral7b_80gb \
trainer.skip_sanity_check=false \
ckpt_path=${ckpt_path} \
trainer.eval_molpo=false \
trainer.eval_modality_util=false \

