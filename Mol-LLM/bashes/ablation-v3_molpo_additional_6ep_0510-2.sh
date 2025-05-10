export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
max_epochs=6
total_batch_size=512
filename=ablation-v3_molpo_additional_6ep_0510-2
pretrained_ckpt_path="'/data/all_checkpoints/ablation-v3_molpo_additional_6ep_0510/epoch=05-step=13157.ckpt'"

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${filename} \
data.data_tag=ablation-v3_0510-molpo-replace-0.3 \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path=${pretrained_ckpt_path} \
trainer.min_lr=0.000005 \
trainer.init_lr=0.00001 \
trainer.warmup_lr=0.000001
