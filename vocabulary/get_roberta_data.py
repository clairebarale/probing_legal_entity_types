import json
import glob
import os
from pathlib import Path
import pprint
from datasets import load_dataset


pp = pprint.PrettyPrinter(indent=4)

def read_json_files(source_folder):
    source_folder = Path(source_folder).expanduser()
    json_files = glob.glob(f"{source_folder}/*.json")
    return json_files

def gen_list(start: int, end: int) -> list:
    mylist = []
    for i in range(start, end):
        mylist.append(i)
    return mylist



def get_cc_news_data(cc_news_path):
    """
    inputs the folder in news, and returns a txt file
    """
    print("---------------------------------------------------------------")
    print("we get cc_news articles in english")
    txt_file = open("data_plms/mysample/text_cc_news.txt","a") # to write the selected articles' text
    mylist = []

    art_en = 0 # counter for the articles in english

    pathlist = Path(cc_news_path).rglob('*') # the list of subdirectories
    for path in pathlist:
        articles = read_json_files(path)
        for art in articles:
            art = open(art)
            text_json = json.load(art)
            if text_json['maintext'] and text_json['language'] == 'en':
                art_en += 1
                mylist.append(text_json['maintext'])
  
    print(f"We collected text from {art_en} articles in english from cc_news common crawl")
    print(len(mylist))

    txt_file.writelines(mylist) # writes the items of a list to the file
    txt_file.close()
    print("---------------------------------------------------------------")

def get_openwebtext_data(dataset):
    mylist = []
    txt_file = open("data_plms/mysample/text_openwebtext.txt","a")
    # input the dataste and return a text file
    print("we get the openweb text data")
    print("---------------------------------------------------------------")
    shuffled_dataset = dataset.shuffle(seed=42)
    dataset_sample = list(shuffled_dataset.take(11875))
    for example in dataset_sample:
         mylist.append(example['text'])
    txt_file.writelines(mylist)
    txt_file.close()
    print("---------------------------------------------------------------")

def get_ccstories_data(source_dataset):
    print("---------------------------------------------------------------")
    mylist = []
    txt_file = open("data_plms/mysample/text_cc_stories.txt","a")
    # input the dataste and return a text file
    print("we get the cc stories text data")
    print("---------------------------------------------------------------")
    dataset = load_dataset("text", data_files={"train": [source_dataset]})
    shuffled_dataset = dataset.shuffle(seed=42)

    sampled_dataset = shuffled_dataset["train"].select(gen_list(0,9688)) # we take 9688 sample
    print(sampled_dataset)
 
    for example in sampled_dataset:
            mylist.append(example['text'])
    print(len(mylist))
    txt_file.writelines(mylist)
    txt_file.close()
    print("---------------------------------------------------------------")


def get_book_corpus_data(source1, source2):
    mylist = []
    txt_file = open("data_plms/mysample/text_bookcorpus.txt","a")
    # input the datas and return a text file
    print("we get the book corpus text data")
    print("---------------------------------------------------------------")
    dataset = load_dataset("text", data_files={"train": [source1, source2]})
    shuffled_dataset = dataset.shuffle(seed=42)
    print(shuffled_dataset)
    sampled_dataset = shuffled_dataset["train"].select(gen_list(0,5000)) # we take 5000 samples, 10% of roberta pt data
    print(sampled_dataset)
 
    for example in sampled_dataset:
            mylist.append(example['text'])
    print(len(mylist))

    txt_file.writelines(mylist)
    txt_file.close()
    print("---------------------------------------------------------------")




if __name__ == "__main__":
    print("---------------------------------------------------------------")
    print("We build a folder with Roberta/Deberta pretraining data (similar to)")
    print("the folder is located in data_plms/")
    print("---------------------------------------------------------------")
    
    # cc_news from common crawl
    cc_news_path = "data_plms/cc_download_articles"
    get_cc_news_data(cc_news_path) # creates a text file

    # open web text
    openweb_ds = load_dataset("Skylion007/openwebtext", streaming=True, split="train") # on huggingface
    get_openwebtext_data(openweb_ds)

    # cc_stories
    cc_stories = "data_plms/cc-stories.txt" # local file downloaded from hugginface
    get_ccstories_data(cc_stories)

    # book corpus
    book_corpus_path1 = "data_plms/books_large_p1.txt"
    book_corpus_path2 = "data_plms/books_large_p2.txt"
    get_book_corpus_data(book_corpus_path1, book_corpus_path2)