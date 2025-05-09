export TOKENIZERS_PARALLELISM=false;
gpus=$1
max_epochs=1
total_batch_size=256

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=ablation-full-tasks_qformer_pretraining_1ep_0509 \
data.data_tag=ablation-full-tasks_0509 \
trainer=mistral7b_80gb_llava_pretraining \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=true \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path="'/data/all_checkpoints/ablation-full-tasks_0509_string_only_gine_custom_0509/last.ckpt'"

