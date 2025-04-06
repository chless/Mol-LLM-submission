
export TOKENIZERS_PARALLELISM=false;
file_name='ood_from-string_only_step2430'
gpus="'1,2,3,4,5,6,7'"

python Mol-LLM/stage3.py \
trainer.devices=$gpus \
filename=$file_name \
mode=test \
data=multi_task \
data.data_tag=ood \
data.raw_data_root=/data/data/Mol-LLM-v7.1 \
gnn=moleculeSTM \
gnn.graph_encoder_ckpt=/data/all_checkpoints/MoleculeSTM/molecule_model.pth \
trainer=mistral7b_80gb_mdpo-v2 \
trainer.selfies_token_path=Mol-LLM/model/selfies_dict.txt \
trainer.logging_dir=/data/all_checkpoints \
trainer.mol_representation=string+graph \
trainer.train_molpo=false \
trainer.eval_molpo=false \
trainer.inference_batch_size=12 \
ckpt_path="'/data/all_checkpoints/MT_from-string_only_0211/epoch=00-step=2430.ckpt'"
