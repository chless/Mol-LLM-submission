export TOKENIZERS_PARALLELISM=false;
gpus=$1
modality=$2
gnn=$3
max_epochs=6
total_batch_size=256

tasks=(
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap"
)

echo "==============Executing task: ablation==============="
python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=ablation-qm9_${modality}_${gnn}_0509 \
data.data_tag=ablation-qm9_0509 \
gnn=${gnn} \
trainer=mistral7b_80gb \
trainer.max_epochs=${max_epochs} \
trainer.mol_representation=${modality} \
trainer.skip_sanity_check=false \
trainer.total_batch_size=${total_batch_size}

