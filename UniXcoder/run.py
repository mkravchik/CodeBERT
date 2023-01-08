import torch
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax
from tqdm import tqdm
import shutil
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_closest_label(model, code, labels):
    tokens_ids = model.tokenize([code],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    _, code_embedding = model(source_ids)
    norm_code_embedding = torch.nn.functional.normalize(code_embedding, p=2, dim=1)

    sims = []
    for label in labels:
        label_tokens_ids = model.tokenize([label],max_length=512,mode="<encoder-only>")
        label_source_ids = torch.tensor(label_tokens_ids).to(device)
        _, nl_embedding = model(label_source_ids)
        norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)

        code_label_similarity = torch.einsum("ac,bc->ab",norm_code_embedding,norm_nl_embedding)

        # print(f'Label {label} similarity {code_label_similarity}')
        sims.append(code_label_similarity.detach().cpu().numpy()[0][0])
    probs = softmax(sims)
    return labels[np.argmax(sims)], np.max(probs), norm_code_embedding.detach().cpu().numpy()

def summarize_code(model: UniXcoder, code):
    context = "/* Belongs to <mask0> (crypto, network, ui) project */ "+code
    tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
    predictions = model.decode(prediction_ids)
    return [x.replace("<mask0>","").strip() for x in predictions[0]]

def display(df_vectors : np.array, labels, name):
    pca = PCA(n_components=2)
    X = df_vectors

    pca = pca.fit(X)
    projected =  pca.transform(X)
    print(X.shape)
    print(projected.shape)

    df = pd.DataFrame(dict(labels=labels, pca0=projected[:, 0], pca1 = projected[:, 1]))

    sns.scatterplot(data=df, x='pca0', y='pca1', hue='labels')
    plt.xlim(-7, 7)
    plt.ylim(-2.5, 7)
    plt.savefig("PCA_%s.pdf"%name, dpi=600, bbox_inches='tight')
    plt.close()

def classify_code(model: UniXcoder, code_jsonl: str, labels, code_tag="code", label_tag="label", labels_map=None, summarize=False):
    true_labels = []
    pred_labels = []
    embeddings = []
    num_lines = sum(1 for line in open(code_jsonl))
    if summarize:
        f_sum = open(code_jsonl+"_predictions.txt", 'w')
    for line in tqdm(open(code_jsonl), total=num_lines):
        try:
            tagged_code = json.loads(line)
            code = ''.join(tagged_code[code_tag]).replace('\n',' ')

            label, prob, emb = find_closest_label(model, code, labels)
            pred_labels.append(label)
            embeddings.append(emb.flatten())
            true_label = tagged_code[label_tag]
            if labels_map is not None:
                true_label = labels_map[true_label]
            true_labels.append(true_label)

            #Write summary
            if summarize:
                tagged_code["summary"]=" OR ".join(summarize_code(model, code))
                json_s = json.dumps(tagged_code)
                f_sum.write(json_s + '\n')

        except Exception as e:
            print("Skipping invalid line:", line, ".", e)

    if summarize:
        f_sum.close()
    
    # display(np.array(embeddings), true_labels, code_jsonl)

    print(f'Got {len(true_labels)} true_labels {true_labels[:2]} ... \nand {len(pred_labels)} pred_labels {pred_labels[:2]}')
    print(confusion_matrix(true_labels, pred_labels ))
    print(classification_report(true_labels, pred_labels, zero_division=0))

code = "  d = _state[3];\n  e = _state[4];\n  #ifdef _SHA1_UNROLL\n  RX_5(R0, 0); RX_5(R0, 5); RX_5(R0, 10);\n  #else\n  int i;\n  for (i = 0; i < 15; i += 5) { RX_5(R0, i); }\n  #endif\n  RX_1_4(R0, R1, 15);\n  #ifdef _SHA1_UNROLL\n"
labels = ["crypto", "network", "ui"]

labels_map = {0:"crypto", 1:"network", 2:"ui"}

for model in ["microsoft/unixcoder-base", "microsoft/unixcoder-base-nine","microsoft/codebert-base"]:
    print(f'Using model {model}')
    model_base = UniXcoder(model)
    model_base.to(device)
    total_params = sum(p.numel() for p in model_base.parameters())
    trainable_params =  sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f'There are {total_params} parameters with {trainable_params} of them trainable')

    # print(f'This looks like {find_closest_label(model_base, code, labels)[0]}')
    classify_code(model_base, "test.jsonl", labels=labels, labels_map=labels_map)
