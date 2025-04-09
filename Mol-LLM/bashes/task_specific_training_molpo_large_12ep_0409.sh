export TOKENIZERS_PARALLELISM=false;
gpus=$1
gnn=$2
modality=string+graph
task=$3
rejected_lambda=$4
molpo_batch_division=$5
graph_encoder_ckpt=$6
projector_type=qformer
max_epochs=12
total_batch_size=88
trained_tokengt_ckpt=/data/all_checkpoints/Custom_gnn_models/TokenGT/best-model.ckpt
moleculestm_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth

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
filename=${task}_molpo_rej-${rejected_lambda}_${gnn}_12ep_0409 \
data.data_tag=${task}_0405_molpo \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=${gnn} \
gnn.graph_encoder_ckpt=${graph_encoder_ckpt} \
trainer=mistral7b_80gb_molpo \
trainer.max_epochs=${max_epochs} \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=${modality} \
trainer.projector_type=${projector_type} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
trainer.rejected_lambda=${rejected_lambda} \
trainer.molpo_batch_division=${molpo_batch_division} \
trainer.every_n_epochs=0

