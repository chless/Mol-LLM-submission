export TOKENIZERS_PARALLELISM=false;
file_name='PP_mistral7b_string+graph_gnn_freeze_1224'
gpus="'0,1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
data.tasks="['bace', 'tox21', 'toxcast', 'smol-property_prediction-bbbp', 'smol-property_prediction-clintox', 'smol-property_prediction-hiv', 'smol-property_prediction-sider', 'qm9_homo', 'qm9_lumo', 'qm9_homo_lumo_gap', 'qm9_additional_label', 'smol-property_prediction-esol', 'smol-property_prediction-lipo']"

