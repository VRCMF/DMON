# --train_file /cw/liir_code/NoCsBack/sun/argument_mining/data/graph/merge_train_neoplasm_196.csv \
# --validation_file /cw/liir_code/NoCsBack/sun/argument_mining/data/graph/merge_test_neoplasm.csv \

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
python /cw/liir_code/NoCsBack/sun/argument_mining/run_graph.py \
--model_name_or_path michiyasunaga/LinkBERT-base \
--train_file /cw/liir_code/NoCsBack/sun/argument_mining/data/cdcp/graph/train.csv \
--validation_file /cw/liir_code/NoCsBack/sun/argument_mining/data/cdcp/graph/test.csv  \
--learning_rate 2e-5 \
--num_train_epochs 10000 \
--max_seq_length 128 \
--output_dir /cw/liir_code/NoCsBack/sun/argument_mining/log/graph_test \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps 2 \
--model_mode 'bert_self' \
--dataset_domain 'cdcp' \
--win_size 13 \
--description 'card' \
--voter_branch "dual"

