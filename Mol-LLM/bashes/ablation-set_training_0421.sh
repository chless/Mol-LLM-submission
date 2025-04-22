export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=$2
gnn=$3
task=ablation
max_epochs=6
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
filename=${task}_${modality}_${gnn}_0421 \
data.data_tag=${task}_0411 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=${gnn} \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=${modality} \
trainser.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size}

