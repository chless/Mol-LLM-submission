export TOKENIZERS_PARALLELISM=false;
gpus=$1
projector_type=$2
modality=$3
max_epochs=12

tasks=(
    "chebi-20-mol2text"
    "reagent_prediction"
    "qm9_homo"
    "forward_reaction_prediction"
)

for task in "${tasks[@]}"; do
    echo "==============Executing task: $task==============="
    python Mol-LLM/stage3.py \
    mode=test \
    trainer.devices=$gpus \
    filename=${task}_${projector_type}_${modality}_12ep_test_0318 \
    data.data_tag=${task}_0219 \
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
    trainer.every_n_epochs=0 \
    ckpt_path=/data/all_checkpoints/${task}_${projector_type}_${modality}_12ep_0318/last.ckpt
done

