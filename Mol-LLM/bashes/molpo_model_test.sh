export TOKENIZERS_PARALLELISM=false;
gpus=$1
data_tag=$2
filename=test_${data_tag}_molpo_0510
ckpt_path="'/data/all_checkpoints/MT_from-string_only_12ep_molpo-wo-L-rej-margin_clip_scale-1.0_6ep_0502_continue/epoch=00-step=1614.ckpt'"

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
data.data_tag=${data_tag} \
trainer=mistral7b_80gb \
trainer.skip_sanity_check=false \
ckpt_path=${ckpt_path}

