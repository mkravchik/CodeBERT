import argparse
import json
import pickle
import contextlib
import sys

import run

class CodeBERTBoost:
    def __init__(self):
        pass

    def fit(self, data):
        @contextlib.contextmanager
        def set_train_args():
            try:
                sys._argv = sys.argv
                sys.argv=[sys.argv[0], "--output_dir","./saved_models",
                "--tokenizer_name","microsoft/unixcoder-base-nine",
                "--model_name_or_path","microsoft/unixcoder-base-nine",
                "--do_train", "--train_data_file","./train_cb.jsonl", "--num_train_epochs", "1", "--block_size", "256",
                "--train_batch_size", "4", "--eval_batch_size", "4", "--learning_rate", "2e-5", "--max_grad_norm",
                "1.0", "--seed", "123456"]
                yield
            finally:
                sys.argv = sys._argv

        with set_train_args():
            run.main()
        pass

    def predict(self, data):
        @contextlib.contextmanager
        def set_test_args():
            try:
                sys._argv = sys.argv
                sys.argv=[sys.argv[0], "--output_dir","./saved_models",
                "--tokenizer_name","microsoft/unixcoder-base-nine",
                "--model_name_or_path","microsoft/unixcoder-base-nine",
                "--do_test", "--train_data_file","./train_cb.jsonl", "--test_data_file", "./test_cb.jsonl",
                "--block_size", "256", "--eval_batch_size", "4", "--seed", "123456"]
                yield
            finally:
                sys.argv = sys._argv

        #TODO convert to the CodeBERT format
        with set_test_args():
            result = run.main()
        #TODO Process the result (we need to translate it back to the labels)

        pass

if __name__ == "__main__":
    clf = CodeBERTBoost()
    # Open the jsonl file
    data = []
    with open('refined_dataset.jsonl', 'r') as f:
        # Iterate over the lines in the file
        for line in f:
            # Parse the line as JSON
            data.append(json.loads(line))
    # Train the model
    # clf.fit(data)
    clf.predict(data)
    # Store the trained model on the disk
    # file_to_store = open("./Boosting/trained_RF.pickle", "wb")
    # pickle.dump(clf, file_to_store)
