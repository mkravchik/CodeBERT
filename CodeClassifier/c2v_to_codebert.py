import json
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--from_file", help="code2vec jsonl")
    parser.add_argument("-t", "--to_file", help="codeBERT jsonl")
    parser.add_argument("-l", "--labels", help="Order of labels", default="crypto,network,GUI")

    args = parser.parse_args()
    print(args)
    labels = args.labels.split(",")

    num_lines = sum(1 for line in open(args.from_file))
    with open(args.to_file, 'w') as f_to:
        for line in tqdm(open(args.from_file), total=num_lines):
            try:
                c2vDict = json.loads(line)
                cbDict = {"code":c2vDict["snippet"], "label":labels.index(c2vDict["label"])}
                json_s = json.dumps(cbDict)
                f_to.write(json_s + '\n')
            except Exception as e:
                print("Skipping invalid line:", line, ".", e)


