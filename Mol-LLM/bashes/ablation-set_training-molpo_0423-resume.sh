export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3,4,5,6,7'"
tag=""
reject_label_mask=false
modality=string+graph
gnn=gine_tokengt
replace_ratio=0.3
task=ablation
max_epochs=6
total_batch_size=256

if [ -n "$tag" ]; then
    filename=${task}_molpo_reject-v2_${tag}_0423
else
    filename=${task}_molpo_reject-v2_0423
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

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${filename} \
data.data_tag=molpo-v2-ablation \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=${gnn} \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=true \
trainer.total_batch_size=${total_batch_size} \
ckpt_path="'/data/all_checkpoints/ablation_molpo_reject-v2_0423/epoch=02-step=4872.ckpt'" \
trainer.reject_label_mask=${reject_label_mask} \
wandb_id=40n4nfe7

