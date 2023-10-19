import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')

def concatenate_csv_files(csv_files, output_file):
    """
    Concatenate csv files into one file.
    :param csv_files: list of csv files
    :param output_file: output file
    :return: None
    """
    with open(output_file, 'w+') as outfile:
        for fname in csv_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def replace_value_by_nan(df, value, replacement):
    """
    Replace a value by NaN in a dataframe.
    :param df: dataframe
    :param value: value to replace
    :param replacement: replacement
    :return: dataframe
    """
    df.replace(to_replace=value, value=replacement, inplace=True)
    return df

def count_values_per_columns(df):
    """
    Count values per columns in a dataframe.
    :param df: dataframe
    :return: None
    """
    for col in df.columns:
        print(col, df[col].count())


# concatenate_csv_files(["annotations_entities_asylex/test.csv", "annotations_entities_asylex/train.csv"], output_file="annotations_entities_asylex/annotations_entities_asylex.csv")

input_file = "annotations_entities_asylex/annotations_entities_asylex.csv"
output_file = "entities_asylex_prompts.csv" # context (sentences), entity type, prompt with masked entity

df = pd.read_csv(input_file, sep=";")
df.drop(columns=["Unnamed: 0", "decisionID"], inplace=True, axis=1)
df = replace_value_by_nan(df, "['']", np.nan)

print("---------------------------------------------------")
print(len(df))
count_values_per_columns(df)
print("---------------------------------------------------")

#Text 1781242
#GPE 573197
#DATE 455703
#NORP 132118
#ORG 346493
#LAW 215296
#CLAIMANT_EVENTS 1
#CREDIBILITY 338451
#DETERMINATION 85524
#CLAIMANT_INFO 128068
#PROCEDURE 410525
#DOC_EVIDENCE 527273
#EXPLANATION 276410
#LEGAL_GROUND 61988
#LAW_CASE 99782
#LAW_REPORT 24777

ENTITY_TYPES = [
    "GPE", 
    "DATE", 
    "NORP", 
    "ORG", 
    "LAW", 
    "CREDIBILITY", 
    "DETERMINATION", 
    "CLAIMANT_INFO", 
    "PROCEDURE", 
    "DOC_EVIDENCE", 
    "EXPLANATION", 
    "LEGAL_GROUND", 
    "LAW_CASE", 
    "LAW_REPORT"]

print("We have {} entity types".format(len(ENTITY_TYPES)))

print("---------------------------------------------------")
print("Creating prompts...")

def clean_col_df_str_entities(str_entities):
    """
    Clean a column of a dataframe.
    :param df: string of entities, list like
    :return: list of entities, type list
    """
    list_entities = str_entities.replace("['", "").replace("']", "").replace("' '", ",").split(",")   
    return list_entities

# for prompt template 1
def mask_entity_in_prompt(sentence, list_entities):
    """
    Replace entity in prompt by <MASK>.
    :param context: a sentence
    :param entity: list of entities
    :return: prompt wiht masked entity
    """
    # we mask the entity in the context and create the column "prompt"
    print("We mask one entity per sentence")
    for entity in list_entities:
        prompt = sentence.replace(entity, "<MASK>")
    return prompt

def get_synonyms(token) -> list:
    """
    :return: tuples containing most similar strings, depending on the defined parameters of similarity
    output looks like this:
    ('machine learning', 'NOUN'), 0.8986967)
    ((word, sense), score) where sense is the POS tag
    """
    synonyms = wn.synonyms(token)
    return synonyms



#########################################################
#########################################################
# PROMPT TEMPLATE 1 (entity in the middle of the sentence)

print("---------------------------------------------------")
print("Creating prompts for template 1...")

for entity_type in tqdm(ENTITY_TYPES):
    # we create prompts for each entity type, i.e. we mask the entity in the context
    # creates one file per entity type
    print("Creating prompts for entity type {}".format(entity_type))
    
    df_prompt2 = df[df[entity_type].notnull()]
    print("We have {} rows for entity type {}".format(len(df_prompt2), entity_type))
    df_prompt2 = df_prompt2[["Text", entity_type]]
    df_prompt2.columns = ["context", "entity"]

    # we clean the column of entities, transform it into a list of entities per row
    df_prompt2["entity"] = df_prompt2["entity"].apply(clean_col_df_str_entities)

    df_prompt2["prompt"] =  df_prompt2["context"]
    df_prompt2.reset_index(inplace=True, drop=True)
    print(df_prompt2.head())


    # df_entity_type["context"] is a col of strings
    # we mask the entity in the context and create the column "prompt"
    for i in range(len(df_prompt2)):
        df_prompt2["prompt"][i] = mask_entity_in_prompt(df_prompt2["context"][i], df_prompt2["entity"][i])
    
    print(df_prompt2.head())
    
    # we save the dataframe as a csv file
    df_prompt2.to_csv("entities_asylex_prompts/{}_prompts_masked_with_context_template1.csv".format(entity_type), sep=";", index=False)

#########################################################
# PROMPT TEMPLATE 2 ("Entity is a [masked entity type]

print("---------------------------------------------------")
print("Creating prompts for template 2...")


template = "<Entity> is a <mask>"

# could get synonyms of entity type as well. eg. person could also be "judge, individual, etc."
#extended_entity_type_list = [] # entity type + synonyms
#for entity in ENTITY_TYPES:
#    extended_entity_type_list.append(get_synonyms(entity))
#extended_entity_type_list = [item for sublist in extended_entity_type_list for item in sublist]
#print(extended_entity_type_list)

dict_entity_types = {
    "GPE": ["city", "country", "region", "state", "province", "area", "nation", "land", "republic", "district", "territory", "division", "zone"],
    "DATE": ["date", "day_of_the_month", "appointment", "particular date", "date stamp", "time", "timestamp", "calendar date", "schedule"],
    "NORP": ["nationality", "religious community", "political group", "ethnic groups", "community", "racial group", "party", "faction", "ideological group", "belief community"],
    "ORG": ["tribunal", "firm", "ngo", "company", "corporation", "business", "nonprofit", "association", "charity", "court", "judicial body"],
    "LAW": ["convention", "international convention", "law", "legislation", "legal code", "treaty", "agreement", "protocol", "statute"],
    "CREDIBILITY":  ["plausibility", "authenticity", "integrity", "trustworthiness", "reliability", "credibility", "believability", "credibility", "credibleness"],
    "DETERMINATION":  ["verdict", "result", "resolution", "judgment", "approval", "denial", "decline", "rejection", "approval", "determination", "finding", "conclusion", "decision", "grant", "refusal", "positive decision", "negative decision"],
    "CLAIMANT_INFO": ["data", "employment", "resident", "national", "inhabitant", "information", "gender", "age", "citizen", "citizenship", "sex", "job", "occupation", "profession"],
    "PROCEDURE": ["affidavit", "documentary evidence", "proof", "testimony", "exhibit", "record", "file", "paperwork", "operation", "procedure", "legal procedure", "legal process", "judicial procedure", "legal steps", "judicial process"],
    "DOC_EVIDENCE": ["proof", "evidence", "document", "written document", "written evidence", "written proof", "written record", "written report", "written statement", "written testimony", "written witness statement"],
    "EXPLANATION": ["explanation", "clarification", "interpretation"],
    "LEGAL_GROUND": ["reason", "ground", "legal ground", "justification", "rationale", "foundation", "legal basis", "legal justification"],
    "LAW_CASE": ["citation", "jurisprudence", "case", "law", "case law", "Legal case", "lawsuit", "Legal matter", "legal precedent", "judicial decisions", "legal rulings"],
    "LAW_REPORT": ["country report", "report", "official report", "written report", "ngo report", "national report", "state report", "regional report", "nonprofit report", "non-governmental organization report", "charity report"]
}

for entity_type in tqdm(ENTITY_TYPES):
    # we create prompts for each entity type, i.e. we mask the entity in the context
    # creates one file per entity type
    print("Creating prompts for entity type {}".format(entity_type))
    
    df_prompt2 = df[df[entity_type].notnull()]
    print("We have {} rows for entity type {}".format(len(df_prompt2), entity_type))
    df_prompt2 = df_prompt2[["Text", entity_type]]
    df_prompt2.columns = ["context", "entity"]

    # we clean the column of entities, transform it into a list of entities per row
    df_prompt2["entity"] = df_prompt2["entity"].apply(clean_col_df_str_entities)
    print(df_prompt2.head())

    df_prompt2["prompt"] =  df_prompt2["entity"]
    df_prompt2.reset_index(inplace=True, drop=True)

    # if prompt contain a list of entities, we create a prompt for each entity
    # if prompt contains only one entity, we create one prompt
    for i in range(len(df_prompt2)):
        if len(df_prompt2["prompt"][i]) > 1:
            for j in range(len(df_prompt2["prompt"][i])):
                df_prompt2["prompt"][i][j] = template.replace("<Entity>", df_prompt2["prompt"][i][j])
        else:
            df_prompt2["prompt"][i] = template.replace("<Entity>", df_prompt2["prompt"][i][0])

    # we save the dataframe as a csv file
    df_prompt2.to_csv("entities_asylex_prompts/{}_prompts_masked_template2.csv".format(entity_type), sep=";", index=False)







