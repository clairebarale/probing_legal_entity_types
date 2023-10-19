from tqdm import tqdm
from pathlib import Path

ds_paths = [
    "data_plms/tokenized_samples/casehold_tokenized.txt",
    "data_plms/tokenized_samples/lexlm_tokenized.txt",
    "data_plms/tokenized_samples/pol_tokenized.txt",
    "data_plms/tokenized_samples/roberta_pt_tokenized.txt"
]

def extract_text(txt_file, dataset_name) -> str:
    # extract text as a string from the txt file
    # count number of lines in the file
    with open(txt_file, 'r') as fp:
        # read as string:
        text = Path(txt_file).read_text()
        return text

output_folder = "data_plms/tokenize_sample_10K_common"

def get_most_common_10000_tokens(text: str) -> list:
    count = {} # dict to store count and token
    
    for token in tqdm(text.split(",")):
        if token in count:
            # existing in the dict already
            count[token] = count[token] + 1
        else:
            # new
            count[token] = 1
    
    count = sorted(count, key=count.get, reverse=True)[:10000]
    return count

for ds in ds_paths:
    ds_name = ds.split("/")[2][:-4]
    text = extract_text(ds, ds_name)
    print("------------------------------------------------")
    print(f'Number of tokens in {ds_name}:', len(text))
    print("------------------------------------------------")
    tokens = get_most_common_10000_tokens(text)
    print(len(tokens))
    
    with open(f'{output_folder}/{ds_name}_tokenized_common.txt', 'a') as outfile:
        outfile.write(",".join(tokens))
    outfile.close()

