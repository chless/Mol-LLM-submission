export TOKENIZERS_PARALLELISM=false;
gpus='0,1,2,3'
batch_size=32
inference_batch_size=3200

python3 MolCA/stage3.py \
--devices $gpus \
--root multi_task \
--mode test \
--filename instruction_tuning_string+graph_ours_galac_0701 \
--stage2_path 'MolCA/all_checkpoints/instruction_tuning_string+graph_ours_galac_0701/step=109000-total_loss=0.990.ckpt' \
--opt_model 'facebook/galactica-1.3b' \
--prompt '[START_I_SMILES]{}[END_I_SMILES].' \
--tune_gnn \
--llm_tune lora \
--batch_size $batch_size \
--inference_batch_size $inference_batch_size \
--val_check_interval 0.1 \
--max_epochs 10 \
--mol_representation string+graph \
--num_beam 1 \
--save_every_n_epochs 1 \
--raw_data_root MolCA/data/multi_task_dataset