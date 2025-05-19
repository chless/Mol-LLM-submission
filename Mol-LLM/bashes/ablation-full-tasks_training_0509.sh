export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=string_only
max_epochs=3
total_batch_size=256

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=ablation-full-tasks_${modality}_0509 \
data.data_tag=ablation-full-tasks_0509 \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size}

