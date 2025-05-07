export TOKENIZERS_PARALLELISM=false;
gpus=$1
gnn=$2
qformer_pretraining=$3
max_epochs=6
total_batch_size=256
filename=ablation_molpo-qformer_pretraining-${qformer_pretraining}_${gnn}_additional_6ep_0507
if [ "$qformer_pretraining" -eq 1 ]; then
    pretrained_ckpt_path="'/data/all_checkpoints/ablation_qformer-${gnn}_pretraining_1ep_0507/last.ckpt'"
else
    pretrained_ckpt_path="'/data/all_checkpoints/ablation_string_only_gine_custom_0421/epoch=05-step=15653.ckpt'"
fi

tasks=(
    "smol-property_prediction-bbbp"
    "smol-property_prediction-clintox"
    "bace"
    "smol-property_prediction-esol"
    "smol-property_prediction-lipo"
    "smol-property_prediction-hiv"
    "smol-property_prediction-sider"
    "chebi-20-mol2text"
    "reagent_prediction"
    "qm9_homo"
    "forward_reaction_prediction"
)

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=filename \
data.data_tag=ablation_0411_molpo-replace-0.3 \
gnn=${gnn} \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
pretrained_ckpt_path=${pretrained_ckpt_path}

