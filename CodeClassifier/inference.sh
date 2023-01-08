python ./run.py \
    --output_dir=./saved_models \
    --tokenizer_name=microsoft/unixcoder-base-nine \
    --model_name_or_path=microsoft/unixcoder-base-nine \
    --do_test \
    --train_data_file=./train_cb.jsonl \
    --test_data_file=./test_cb.jsonl \
    --num_train_epochs 1 \
    --block_size 64 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee test.log