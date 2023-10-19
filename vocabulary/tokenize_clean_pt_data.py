print("program has started...")

from tqdm import tqdm
import glob
from pathlib import Path
import re
from joblib import Parallel, delayed
import nltk
import nltk.data
from nltk.corpus import stopwords
import os
import spacy
from spacy.lang.en import English

nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

print("program has loaded py modules...")
print("loaded spacy")

stopw = set(stopwords.words('english'))  # build-in list of stop words
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

source_dir = "data_plms/mysample"

files_list = glob.glob(f"{source_dir}/*.txt")

# get Roberta files list
# roberta_files_list = []
# source_folder = Path(source_dir).expanduser()
# roberta_files_list = glob.glob(f"{source_folder}/text_*.txt")
# print(roberta_files_list)

# combine the .txt files 
# with open(f'{source_folder}/roberta_pt.txt', 'a') as outfile:
#    for fname in roberta_files_list:
#        print(fname)
#        with open(fname) as infile:
#            for line in infile:
#                outfile.write(line)

# output_folder = "data_plms/tokenized_samples/" # to store the output files 
output_folder = "data_plms/tokenized_samples/country/" # to store the output files 

ds_paths = [
    #"data_plms/mysample/roberta_pt.txt",
    #"data_plms/mysample/casehold.txt",
    #"data_plms/mysample/lexlm.txt",
    #"data_plms/mysample/pol.txt"
    "data_plms/mysample/country/lexlm_can.txt",  ### LexLM by country
    "data_plms/mysample/country/lexlm_eu.txt",
    "data_plms/mysample/country/lexlm_ind.txt",
    "data_plms/mysample/country/lexlm_uk.txt",
    "data_plms/mysample/country/lexlm_us.txt",
]



def extract_text(txt_file, dataset_name) -> str:
    # extract text as a string from the txt file
    # count number of lines in the file
    with open(txt_file, 'r') as fp:
        lines = len(fp.readlines())
        print("------------------------------------------------")
        print(f'Total Number of lines in {dataset_name}:', lines)
        # read as string:
        text = Path(txt_file).read_text()
        return text

def clean_text(text: str) -> str:
    clean = re.sub('\n', '', text) # newlines
    clean = re.sub('\W', '', clean) # any non word char
    clean = re.sub('\d+', '', clean) # any digits
    clean = re.sub('\s+', '', clean) # any whitespace/s char
    return clean

def process(sentence: str):
    # tokenizer
    # the tokenizer takes a string of text and turns it into a Doc
    return nlp.pipe(sentence, disable=["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "textcat", "lemmatizer"], n_process=-1) 

def extract_tokens(text: str) -> list:
    # nltk.tokenize.punkt module, sentence tokenizer that works on legal citations
    sentences = sent_detector.tokenize(text.strip())
    print("sentence tokenization done!")
    
    list_tokens = []
    for doc in tokenizer.pipe(sentences): 
        list_tokens.append([t.text for t in doc])
    
    # docs  = Parallel(n_jobs=-1)(delayed(process)(sentences[i]) for i in range(len(sentences)))
    # for i in range(len(sentences)): 
        # Iterate through each token (t) in the doc object 
    #    list_tokens.append([t.text for t in docs[i]])
    
    list_tokens = [item for sublist in list_tokens for item in sublist]
    return list_tokens

def clean_tokens(tokens: list) -> list:
    list_clean_tokens = []
    for token in tokens:
        list_clean_tokens.append(clean_text(token))

    # lowercase, remove stopwords and empty strings:
    list_clean_tokens = [x.lower() for x in list_clean_tokens if x and x not in stopw]
    return list_clean_tokens



for ds in ds_paths:
    ds_name = ds.split("/")[2][:-4]
    text = extract_text(ds, dataset_name=ds_name)

    tokens = extract_tokens(text) # returns a list of tokens
    print("tokenization done!")
    tokens_cleaned = clean_tokens(tokens)

    with open(f'{output_folder}/{ds_name}_tokenized.txt', 'a') as outfile:
        outfile.write(",".join(tokens_cleaned))
    outfile.close()
