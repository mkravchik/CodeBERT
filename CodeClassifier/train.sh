python ./run.py \
    --output_dir=./saved_models \
    --tokenizer_name=neulab/codebert-cpp \
    --model_name_or_path=neulab/codebert-cpp \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=./train_cb.jsonl \
    --eval_data_file=./valid_cb.jsonl \
    --test_data_file=./test_cb.jsonl \
    --num_train_epochs 1 \
    --block_size 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee train.log

    #     --tokenizer_name=microsoft/codebert-base \
    # --model_name_or_path=microsoft/codebert-base \
    # --tokenizer_name=microsoft/unixcoder-base-nine \
    # --model_name_or_path=microsoft/unixcoder-base-nine \
