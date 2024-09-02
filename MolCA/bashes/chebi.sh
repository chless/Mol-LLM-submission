{
export TOKENIZERS_PARALLELISM=false;

python stage2.py --root 'data/ChEBI-20_data' --devices '0,1' --filename "ft_chebi"  --llm_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --tune_llm lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10  --stage2_path 'path-to-stage2-ckpt';
exit
}