export TOKENIZERS_PARALLELISM=false;
gpus=$1
anc_reject_clip=$2
modality=string+graph
gnn=gine_tokengt
max_epochs=3
total_batch_size=256

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

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=ablation_molpo-anc_reject_clip-${anc_reject_clip}_additional-3ep_0424 \
data.data_tag=ablation_0411_molpo-replace-0.3 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=${gnn} \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
trainer.anc_reject_clip=${anc_reject_clip} \
pretrained_ckpt_path="'/data/all_checkpoints/ablation_molpo_reject-v2_0423/epoch=02-step=4872.ckpt'"

