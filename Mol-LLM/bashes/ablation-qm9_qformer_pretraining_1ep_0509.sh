export TOKENIZERS_PARALLELISM=false;
gpus=$1
gnn=$2
max_epochs=1
total_batch_size=256


tasks=(
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap"
)

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=ablation-qm9_qformer-${gnn}_pretraining_1ep_0509 \
data.data_tag=ablation-qm9_0509 \
gnn=${gnn} \
trainer=mistral7b_80gb_llava_pretraining \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=true \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path="'/data/all_checkpoints/ablation-qm9_string_only_gine_custom_0509/last.ckpt'"

