export TOKENIZERS_PARALLELISM=false;
gpus=$1
gnn=$2
max_epochs=1
total_batch_size=256


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

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=ablation-pp_qformer-${gnn}_pretraining_1ep_0508 \
data.data_tag=ablation-pp_0508 \
gnn=${gnn} \
trainer=mistral7b_80gb_llava_pretraining \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=true \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path="'/data/all_checkpoints/ablation-pp_string_only_gine_custom_0508/last.ckpt'"

