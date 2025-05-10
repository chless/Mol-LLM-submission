export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
modality=$1
max_epochs=6
total_batch_size=256

filename=ablation-v3_sft_${modality}_additional_6ep_0510-2
pretrained_ckpt_path="'/data/all_checkpoints/ablation-v3_sft_string+graph_additional_6ep_0510/epoch=05-step=13157.ckpt'"

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${filename} \
data.data_tag=ablation-v3_0510 \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path=${pretrained_ckpt_path} \
trainer.min_lr=0.000005 \
trainer.init_lr=0.00001 \
trainer.warmup_lr=0.000001 \
trainer.val_check_interval=0.20 \
trainer.batch_size=8

