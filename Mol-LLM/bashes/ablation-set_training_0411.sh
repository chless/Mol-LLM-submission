export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=$2
task=ablation
projector_type=qformer
max_epochs=12
total_batch_size=256
gradient_clip_val=0.5
gnn=gine_custom
graph_encoder_ckpt=/data/all_checkpoints/Custom_gnn_models/GINE/best-model.ckpt

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
filename=${task}_${modality}_${gnn}_grad-clip-${gradient_clip_val}_12ep_0411 \
data.data_tag=${task}_0411 \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=${gnn} \
gnn.graph_encoder_ckpt=${graph_encoder_ckpt} \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=${modality} \
trainer.projector_type=${projector_type} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
trainer.gradient_clip_val=${gradient_clip_val} \
trainer.every_n_epochs=0

