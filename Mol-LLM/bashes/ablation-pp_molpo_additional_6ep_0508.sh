export TOKENIZERS_PARALLELISM=false;
gpus=$1
gnn=$2
qformer_pretraining=$3
max_epochs=6
total_batch_size=256
filename=ablation-pp_molpo-qformer_pretraining-${qformer_pretraining}_${gnn}_additional_6ep_0508
if [ "$qformer_pretraining" -eq 1 ]; then
    pretrained_ckpt_path="'/data/all_checkpoints/ablation-pp_qformer-${gnn}_pretraining_1ep_0508/last.ckpt'"
else
    pretrained_ckpt_path="'/data/all_checkpoints/ablation-pp_string_only_gine_custom_0508/last.ckpt'"
fi

tasks=(
    "bace",
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "smol-property_prediction-esol",
    "smol-property_prediction-lipo",
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap"
)


echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${filename} \
data.data_tag=ablation-pp_0508-molpo-replace-0.3 \
gnn=${gnn} \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path=${pretrained_ckpt_path}

