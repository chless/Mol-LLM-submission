export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=$2
gnn=$3
max_epochs=6
total_batch_size=256
if [ "$modality" != "string_only" ]; then
    filename=ablation-qm9_sft_${modality}_${gnn}_additional_6ep_0509
    pretrained_ckpt_path="'/data/all_checkpoints/ablation-qm9_qformer-${gnn}_pretraining_1ep_0509/last.ckpt'"
else
    filename=ablation-qm9_sft_${modality}_additional_6ep_0509
    pretrained_ckpt_path="'/data/all_checkpoints/ablation-qm9_string_only_gine_custom_0509/last.ckpt'"
fi

tasks=(
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap"
)

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${filename} \
data.data_tag=ablation-qm9_0509 \
gnn=${gnn} \
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

