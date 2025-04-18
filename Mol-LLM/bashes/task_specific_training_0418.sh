export TOKENIZERS_PARALLELISM=false;

# small dataset
max_epochs_small=50
total_batch_size_small=32
tasks_small=(
    "smol-property_prediction-bbbp"
    "smol-property_prediction-clintox"
    "bace"
    "smol-property_prediction-esol"
    "smol-property_prediction-lipo"
)

# large dataset
max_epochs_large=12
total_batch_size_large=88
tasks_large=(
    "smol-property_prediction-hiv"
    "smol-property_prediction-sider"
    "chebi-20-text2mol"
    "chebi-20-mol2text"
    "smol-molecule_generation"
    "smol-molecule_captioning"
    "reagent_prediction"
    "qm9_homo"
    "qm9_lumo"
    "qm9_homo_lumo_gap"
    "forward_reaction_prediction"
    "smol-forward_synthesis"
    "retrosynthesis"
    "smol-retrosynthesis"
)

gpus=$1
modality=$2
data=$3
loss=$4

if [ "$loss" == "molpo" ]; then
    trainer="mistral7b_80gb_molpo"
else
    trainer="mistral7b_80gb"
fi

max_epochs=$max_epochs_large
total_batch_size=$total_batch_size_large


echo "==============Executing task: $task | trainer: $trainer=============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${data}_${modality}_${loss}_0418 \
data.data_tag=${data} \
trainer=${trainer} \
trainer.max_epochs=${max_epochs} \
trainer.total_batch_size=$(total_batch_size) \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=false \
trainer.every_n_epochs=0 \
trainer.min_lr=0.00001 \
trainer.init_lr=0.0001 \
trainer.warmup_lr=0.00001 \
trainer.warmup_epochs=0.25

