# inspired from the dont stop pretraining paper

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
from typing import List
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse

def load_data(data_path: str) -> List[str]:
    # input of this function is a json or jsonl file
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {"text": line} # creating a python json object
                text = example['text']
                examples.append(text)
    return examples


def load_vocab(file):
    text = load_data(file)
    count_vectorizer = CountVectorizer() # Pruning: we removed: min_df=3: cut-off = ignore terms that have a document frequency strictly lower than the given threshold
    count_vectorizer.fit(tqdm(text)) # building the vocab from the raw text
    vocab = set(count_vectorizer.vocabulary_.keys())
    return vocab

# if run as a script
parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+", help="files containing tokenized text from each domain", required=True)
parser.add_argument("--output_file", help="path to save heatmap", required=True)
args = parser.parse_args()
files = args.files

vocabs = {}


# gets each file, one by domain (we have one file named file that contains all words)
for file in files:
    if 'roberta' in file:
        key = 'Gen'
    elif 'casehold' in file:
        key = 'CH'
    elif 'pol' in file:
        key = 'PoL'
    elif 'lexlm' in file:
        key = 'LexLM'
    vocabs[key] = load_vocab(file)
    
file_pairs = itertools.combinations(list(vocabs.keys()), 2)
    
overlaps = {}
for x, y in file_pairs:
    intersection = vocabs[x] & vocabs[y]
    union = (vocabs[x] | vocabs[y])
    overlaps[x + "_" + y] = len(intersection) / len(union)
    
data = []

with open("overlaps_without_stopwords", "a") as f:
    json.dump(overlaps, f)


z = {}
for key in overlaps.keys():
    file_1, file_2 = key.split('_')
    if not z.get(file_1):
        z[file_1] = {}
    z[file_1][file_2] = overlaps[key]
    if not z.get(file_2):
        z[file_2] = {}
    z[file_2][file_1] = overlaps[key]

labels = ["Gen", "CH", "PoL", "LexLM"]

# Heatmap 
sns.set(context="paper", style="white", font_scale=2.1)

for ix, key in enumerate(labels):
    items = []
    for subkey in labels:
        if not z[key].get(subkey):
            items.append(1.0)
        else:
            items.append(z[key][subkey])
    data.append(items)
data = np.array(data) * 100
ax = sns.heatmap(data, cmap="Purples", vmin=30, xticklabels=labels, annot=True, fmt=".1f", cbar=False, yticklabels=labels)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(args.output_file, dpi=300)
