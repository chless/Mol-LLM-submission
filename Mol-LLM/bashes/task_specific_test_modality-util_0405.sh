export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=string+graph
task=$2
tag=$3
eval_modality_util=$4
ckpt_path=$5

tasks=(
    "smol-property_prediction-bbbp"
    "smol-property_prediction-clintox"
    "bace"
    "smol-property_prediction-esol"
    "smol-property_prediction-lipo"
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

echo "==============Executing task: $task==============="
echo "==============eval_modality_util: $eval_modality_util==============="
python Mol-LLM/stage3.py \
mode=test \
ckpt_path=${ckpt_path} \
trainer.devices=$gpus \
filename=${task}_test_${eval_modality_util}-random_${tag}_0405 \
data.data_tag=${task}_0219 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=true \
trainer.eval_modality_util=${eval_modality_util}

