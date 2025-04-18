export TOKENIZERS_PARALLELISM=false;
gpus=$1
tag=$2
rejected_lambda=1.2
modality=string+graph
projector_type=qformer
max_epochs=12
total_batch_size=88
gnn=gine_tokengt


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
filename=${tag}_rej-${rejected_lambda}_${gnn} \
data.data_tag=${tag} \
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
trainer.min_lr=0.00001 \
trainer.init_lr=0.0001 \
trainer.warmup_lr=0.00001 \
trainer.warmup_epochs=0.25 \
trainer.every_n_epochs=0

