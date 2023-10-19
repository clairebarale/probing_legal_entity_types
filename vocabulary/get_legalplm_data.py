# we compare 3 legal PLM:
# LexLM large # https://huggingface.co/datasets/lexlms/lex_files # 19 billion tokens
# PoL bert large (henderson 2022) #32 billion tokens, mostly US
# CaseHOLD - CaseBERT base (harvatd law case corpus, Zheng et al 2021)
# all pretraining dataset available on huggingface

from datasets import load_dataset, Features, Value
import pprint
from pathlib import Path

pp = pprint.PrettyPrinter(indent=4)


def get_casehold_data(dataset):
    #pp.pprint(next(iter(dataset))) # an iterable dataset
 
    mylist = []
    txt_file = open(f"data_plms/mysample/casehold.txt","a")
    # input the dataste and return a text file
    
    print("---------------------------------------------------------------")
    print(f"we get the casehold pretraining data")
    
    shuffled_dataset = dataset.shuffle(seed=42)
    dataset_sample = list(shuffled_dataset.take(50000))

    for example in dataset_sample:
         mylist.append(example['citing_prompt'])
         mylist.append(example['holding_0'])
         mylist.append(example['holding_1'])
         mylist.append(example['holding_2'])
         mylist.append(example['holding_3'])
         mylist.append(example['holding_4'])

    print(f"we get {int(len(mylist)/6)} examples for casehold pt data")
    txt_file.writelines(mylist)
    txt_file.close()
    print("---------------------------------------------------------------")

def get_pol_data(dataset):
     # pp.pprint(next(iter(dataset))) # an iterable dataset
 
     mylist = []
     txt_file = open(f"data_plms/mysample/pol.txt","a")
     # input the dataste and return a text file
    
     print("---------------------------------------------------------------")
     print(f"we get the pol pretraining data")
    
     shuffled_dataset = dataset.shuffle(seed=42)
     dataset_sample = list(shuffled_dataset.take(50000))

     for example in dataset_sample:
          mylist.append(example['text'])

     print(f"we get {int(len(mylist))} examples for pol pt data")
     txt_file.writelines(mylist)
     txt_file.close()
     print("---------------------------------------------------------------")


def get_lexlm_data(dataset):

     #pp.pprint(next(iter(dataset))) # an iterable dataset

     mylist = []
     txt_file = open(f"data_plms/mysample/lexlm.txt","a")
     # input the dataste and return a text file
    
     print("---------------------------------------------------------------")
     print(f"we get the lexlm pretraining data")
    
     shuffled_dataset = dataset.shuffle(seed=42)
     dataset_sample = list(shuffled_dataset.take(50000))

     for example in dataset_sample:
          mylist.append(example['text'])

     print(f"we get {int(len(mylist))} examples for lexlm pt data")
     txt_file.writelines(mylist)
     txt_file.close()
     print("---------------------------------------------------------------")


def get_lexlm_country(dataset, country):
     #pp.pprint(next(iter(dataset))) # an iterable dataset
     
     mylist = []
     txt_file = open(f"data_plms/mysample/country/lexlm_{country}.txt","a")
     # input the dataste and return a text file
    
     print("---------------------------------------------------------------")
     print(f"we get the lexlm {country} pretraining data")
    
     shuffled_dataset = dataset.shuffle(seed=42)
     dataset_sample = list(shuffled_dataset.take(10000))

     for example in dataset_sample:
          mylist.append(example['text'])

     print(f"we get {int(len(mylist))} examples for lexlm {country} pt data")
     txt_file.writelines(mylist)
     txt_file.close()
     print("---------------------------------------------------------------")


if __name__ == "__main__":
     
     casehold_ds = load_dataset("casehold/casehold", streaming=True, split = "train")
     get_casehold_data(casehold_ds)

     pol_large_ds = load_dataset("pile-of-law/pile-of-law",'all', streaming=True, split = "train")
     get_pol_data(pol_large_ds)
     
     # lexlm
     DATASET_FILES = ["courtlistener.zip", 
                      "canadian_court_cases.zip", 
                      "canadian_legislation.zip", 
                      "ecthr_cases.zip", 
                      "eurlex.zip", 
                      "indian_courts_cases.zip", 
                      "uk_courts_cases.zip", 
                      "uk_legislation.zip",
                      "us_contracts.zip", 
                      "us_legislation.zip"
     ]

     
     lexlm_ds = load_dataset("lexlms/lex_files", data_files = DATASET_FILES, streaming=True, split="train")
     get_lexlm_data(lexlm_ds)
     
     us_ds = load_dataset('lexlms/lex_files', data_files=["courtlistener.zip", "us_contracts.zip", "us_legislation.zip"], streaming=True, split = "train")
     eu_ds = load_dataset('lexlms/lex_files', data_files=["ecthr_cases.zip", "eurlex.zip"], streaming=True, split="train")
     uk_ds = load_dataset('lexlms/lex_files', data_files=["uk_legislation.zip", "uk_courts_cases.zip"],streaming=True, split = "train")
     can_ds = load_dataset('lexlms/lex_files', data_files=["canadian_court_cases.zip", "canadian_legislation.zip"], streaming=True, split = "train")
     ind_ds = load_dataset('lexlms/lex_files', data_files="indian_courts_cases.zip", streaming=True, split = "train")

     print("datasets loaded")
     
     get_lexlm_country(eu_ds, country="eu")
     get_lexlm_country(uk_ds, country="uk")
     get_lexlm_country(can_ds, country="can")
     get_lexlm_country(ind_ds, country="ind")
     get_lexlm_country(us_ds, country="us")

