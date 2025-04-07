export TOKENIZERS_PARALLELISM=false;
gpus=$1
rejected_lambda=$2
modality=string+graph
projector_type=qformer
task=qm9_homo
max_epochs=12
total_batch_size=88

tasks=(
    "smol-property_prediction-hiv"
    "smol-property_prediction-sider"
    "chebi-20-mol2text"
    "smol-molecule_captioning"
    "reagent_prediction"
    "qm9_homo"
    "forward_reaction_prediction"
    "smol-forward_synthesis"
)

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${task}_molpo_rej-${rejected_lambda}_12ep_0405 \
data.data_tag=${task}_0405_molpo \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=${modality} \
trainer.projector_type=${projector_type} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
trainer.rejected_lambda=${rejected_lambda} \
trainer.every_n_epochs=0

