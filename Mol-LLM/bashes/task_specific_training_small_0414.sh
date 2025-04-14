export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=$2
task=$3
projector_type=qformer
max_epochs=50
total_batch_size=32
gradient_clip_val=0.5
gnn=gine_custom
graph_encoder_ckpt=/data/all_checkpoints/Custom_gnn_models/GINE/best-model.ckpt

tasks=(
    "smol-property_prediction-bbbp"
    "smol-property_prediction-clintox"
    "bace"
    "smol-property_prediction-esol"
    "smol-property_prediction-lipo"
)

echo "==============Executing task: $task==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=${task}_${modality}_${gnn}_0414 \
data.data_tag=${task}_target_value_0414_molpo \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=${modality} \
trainer.projector_type=${projector_type} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size} \
trainer.every_n_epochs=0

