export TOKENIZERS_PARALLELISM=false;
ckpt_path=/data/yuheon/logging
file_name='hug_test_g'
gpus="'0,1,2,3'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
data.raw_data_root=/data/Mol-LLM-v6 \
gnn.graph_encoder_ckpt=MoleculeSTM/molecule_model.pth \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.mol_representation=string+graph \
filename=$ckpt_path \
trainer.logging_dir=${ckpt_path}/${filename}/all_checkpoints/





