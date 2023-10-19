print("loading modules...")
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
print("loading transformers pipeline...")
from transformers import pipeline
print("loading torch...")
import torch
import argparse
import pandas as pd
import os

# Define entity type list
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
    "LAW_CASE": ["citation", "jurisprudence", "case", "law", "case law", "legal case", "lawsuit", "legal matter", "legal precedent", "judicial decisions", "legal rulings"],
    "LAW_REPORT": ["country report", "report", "official report", "written report", "ngo report", "national report", "state report", "regional report", "nonprofit report", "non-governmental organization report", "charity report"]
}


dict_14_entity_types = {
    "GPE": ["location"],
    "DATE": ["date"],
    "NORP": ["nationality", "community", "group"],
    "ORG": ["tribunal", "company"],
    "LAW": ["convention", "law"],
    "CREDIBILITY":  ["credibility"],
    "DETERMINATION":  ["judgment", "determination"],
    "CLAIMANT_INFO": ["employment", "gender", "age", "citizenship", ],
    "PROCEDURE": ["procedure"],
    "DOC_EVIDENCE": ["evidence"],
    "EXPLANATION": ["explanation"],
    "LEGAL_GROUND": ["reason""legal ground"],
    "LAW_CASE": ["case law", "precedent"],
    "LAW_REPORT": ["country report"]
}

def get_list_entities_from_dict(dict_entity_types):
    list_all_entities = []
    for entity_type in dict_entity_types:
        list_all_entities.extend(dict_entity_types[entity_type])
    return list_all_entities

# for 10000 cloze sentences (rows) in the csv file
def get_cloze_sentences_from_csv(csv_file):
    df = pd.read_csv(csv_file, sep=";")
    return df[:10000]


# Predict entity type
def predict_entity_type(cloze_sentence, entity_types, model, tokenizer):
    # Tokenize the sentence
    inputs = tokenizer(cloze_sentence, return_tensors="pt", padding=True, truncation=True, max_length=500, add_special_tokens = True)

    # Pass the tokenized input to the RoBERTa model and retrieve the logits
    with torch.no_grad():
        logits = model(**inputs).logits
    
    
    # retrieve index of <mask>
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    if mask_token_index.numel() > 0:
        masked_token_logits = logits[0, mask_token_index, :]

        # Select the entity type prediction from the provided list
        predicted_entity_id = torch.argmax(masked_token_logits).item()% len(entity_types)

        predicted_entity_type = entity_types[predicted_entity_id]
        # print(f"Predicted entity type: {predicted_entity_type}")
        return predicted_entity_type

        #top_5_tokens = torch.topk(masked_token_logits, 5, dim=1).indices[0].tolist()
        #for token in top_5_tokens:
        #    print(cloze_sentence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    else:
        print("No mask token found in input")

if __name__ == "__main__":
    print("setting env var...")
    print("script name: entity_type_pred_mlm.py") 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    print("setting empty torch cuda cache...")
    torch.cuda.empty_cache() 

    parser = argparse.ArgumentParser()
    print("script started..")

    # required arguments
    parser.add_argument('--in_file', dest='in_file', action='store', required=True, help='the path of the input file')
    parser.add_argument('--device', dest='device', action='store', required=True, help='cpu or cuda')
    args = parser.parse_args()
    args = vars(args)
    in_file = args['in_file'] 
    device = args['device']

    TRANSFORMERS_OFFLINE=1

    # hf path for each model
    Roberta = "roberta-base"
    Debertav3 = "microsoft/deberta-v3-base"
    CaseHOLD = "casehold/legalbert"
    PoL = "pile-of-law/legalbert-large-1.7M-2"
    LexLM = "lexlms/legal-roberta-large"
    
    MODELS = [Roberta, Debertav3, CaseHOLD, PoL, LexLM]
    max_length = 512

    for model_id in MODELS:

        # Load tokenizer and model
        print(f"load {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding='max_length', truncation=True)

        config = AutoConfig.from_pretrained(model_id, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = AutoModelForMaskedLM.from_config(config)
        print("model and tokenizer loaded!")

        ENTITY_TYPES = get_list_entities_from_dict(dict_14_entity_types)

        # get cloze sentences from csv
        df_cloze_sentences = get_cloze_sentences_from_csv(in_file) # a dataframe with on columns of cloze sentences prompts

        # replacing each prompt mask toek with the tokenizer mask token
        df_cloze_sentences = df_cloze_sentences.replace("<MASK>", tokenizer.mask_token, regex=True)
        print(df_cloze_sentences.head())
    
        print("predicting entity type now....")
        # predict entity type for each cloze sentence and write the prediction to a new column
        df_cloze_sentences['output_prediction'] = df_cloze_sentences['prompt'].apply(lambda x: predict_entity_type(x, ENTITY_TYPES, model, tokenizer))
        print(df_cloze_sentences)

        # print to a csv file
        entity_in_file = in_file.split("/")[-1].split(".")[0]
        folder = f"./mlm_cloze_text_entities_experiment/{model_id}/"
        output_filename = f"shortlist_output_{entity_in_file}.csv"
        output_file = os.path.join(folder, output_filename)
        print(output_file)
        df_cloze_sentences.to_csv(output_file, sep=";", index=False)



    
    exit()

    #cloze_sentence = f"The hearing took place in {tokenizer.mask_token}."
    #print(cloze_sentence)
    # will contain the predicted entity type based on your cloze-style sentence
    #prediction = predict_entity_type(cloze_sentence, ENTITY_TYPES, model, tokenizer)
