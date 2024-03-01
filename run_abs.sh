export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
python run_graph.py \
--model_name_or_path michiyasunaga/BioLinkBERT-base \
--train_file 
--validation_file 
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_seq_length 128 \
--output_dir 
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps 1 \
--model_mode 'bert_mtl_1d' \
--dataset_domain 'absRCT' \
--win_size 13 \
--description 'debug' \
--voter_branch "dual" \
--full_map
