export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
modality=$1
max_epochs=6
total_batch_size=256
if [ "$modality" != "string_only" ]; then
    filename=ablation-v3_sft_${modality}_additional_6ep_0510
    pretrained_ckpt_path="'/data/all_checkpoints/ablation-full-tasks_qformer_pretraining_1ep_0510/last.ckpt'"
else
    filename=ablation-v3_sft_${modality}_additional_6ep_0510
    pretrained_ckpt_path="'/data/all_checkpoints/ablation-full-tasks_string_only_0509/epoch=02-step=14561.ckpt'"
fi

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
trainer.init_lr=0.00004 \
trainer.warmup_lr=0.000004 \
trainer.val_check_interval=0.20 \
trainer.batch_size=8

