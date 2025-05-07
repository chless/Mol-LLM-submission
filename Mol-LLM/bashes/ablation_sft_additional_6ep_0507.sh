export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=$2
gnn=$3
max_epochs=6
total_batch_size=256
if [ "$modality" != "string_only" ]; then
    filename=ablation_sft_${modality}_${gnn}_additional_6ep_0507
    pretrained_ckpt_path="'/data/all_checkpoints/ablation_qformer-${gnn}_pretraining_1ep_0507/last.ckpt'"
else
    filename=ablation_sft_${modality}_additional_6ep_0507
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
filename=${filename} \
data.data_tag=ablation_0411 \
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

