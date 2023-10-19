import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, f1_score,recall_score
import argparse

#######################
# predictions
asylex_llama_qa = {
    "GPE": "llama2_qa_entity_types/output_GPE_for_perplexity_measure.csv",
    "DATE": "llama2_qa_entity_types/output_DATE_for_perplexity_measure.csv",
    "NORP": "llama2_qa_entity_types/output_NORP_for_perplexity_measure.csv",
    "ORG": "llama2_qa_entity_types/output_ORG_for_perplexity_measure.csv",
    "LAW": "llama2_qa_entity_types/output_LAW_for_perplexity_measure.csv",
    "CREDIBILITY": "llama2_qa_entity_types/output_CREDIBILITY_for_perplexity_measure.csv",
    "DETERMINATION": "llama2_qa_entity_types/output_DETERMINATION_for_perplexity_measure.csv" ,
    "CLAIMANT_INFO": "llama2_qa_entity_types/output_CLAIMANT_INFO_for_perplexity_measure.csv",
    "PROCEDURE": "llama2_qa_entity_types/output_PROCEDURE_for_perplexity_measure.csv",
    "DOC_EVIDENCE": "llama2_qa_entity_types/output_DOC_EVIDENCE_for_perplexity_measure.csv",
    "EXPLANATION": "llama2_qa_entity_types/output_EXPLANATION_for_perplexity_measure.csv",
    "LEGAL_GROUND": "llama2_qa_entity_types/output_LEGAL_GROUND_for_perplexity_measure.csv",
    "LAW_CASE": "llama2_qa_entity_types/output_LAW_CASE_for_perplexity_measure.csv",
    "LAW_REPORT": "llama2_qa_entity_types/output_LAW_REPORT_for_perplexity_measure.csv"
}

#######################

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

#######################

def clean_prediction_output(df, list_gold_entities):
    # if more that 5 words, consider it as a wrong prediction
    df["prediction"] = df["prediction"].apply(lambda x: x if len(x.split()) <= 5 else "wrong")
    # remove the brackets
    df["prediction"] = df["prediction"].str.replace("[", "")
    df["prediction"] = df["prediction"].str.replace("]", "")
    # lowercase
    df["prediction"] = df["prediction"].str.lower()
    # remove random new lines
    df["prediction"] = df["prediction"].str.replace("\n", "")

    for entity_type in list_gold_entities:
        # replace sentence with entity type if entity type is in the sentence
        # for each row in the dataframe in the prediction col
        # even if it is surrounded by quotes
        df["prediction"] = df["prediction"].apply(lambda x: entity_type if entity_type in x else x)

    return df


#######################

# convert to list
list_entity_types = list(dict_entity_types.values())
# flatten the list
list_entity_types = [item for sublist in list_entity_types for item in sublist]
print(list_entity_types)   

# create dataframe to store all erros per model
df_errors = pd.DataFrame()



print("*** AsyLex dataset ***")

# get the file with prediction adn the list of gold entities per entity type
for ent_type in asylex_llama_qa:
    print(f"***Entity: {ent_type}***")

    df_output_prediction = pd.read_csv(asylex_llama_qa[ent_type], sep=';', usecols=["entity", "prediction"])
  
    list_gold_entities = dict_entity_types[ent_type]
    
    df_output_prediction = clean_prediction_output(df_output_prediction,list_gold_entities)

   
    
    # replace values in the dataframe that are in list by entity type
    # this way whatever the answer, as long as it is a entity type, it will be counted as correct
    # e.g. city will be replaced by GPE, country by GPE, etc. 
    df_output_prediction = df_output_prediction.replace(list_gold_entities, ent_type)
    df_output_prediction["gold_value"] = ent_type
    print(df_output_prediction.head(10))

    # count number of occurences of the entity in the dataframe
    count = df_output_prediction["prediction"].value_counts()[ent_type]
    print(f"Number of occurences of {ent_type}: ", count)
    print("-------------------------------------")


    df_output_prediction.astype(str, errors='ignore')
    df_output_prediction = df_output_prediction.dropna()
    print(len(df_output_prediction))
    
    y_true = df_output_prediction["gold_value"]
    y_pred = df_output_prediction['prediction']

    # Generate the classification report
    #report = classification_report(y_true, y_pred, digits=4)
    #f1 = f1_score(y_true, y_pred, average="weighted")
    #conf_matrix = confusion_matrix(y_true, y_pred,)
    #recall = recall_score(y_true, y_pred, average="weighted")
    #print(report)
    #print("-------------------------------------")
    #print(f"F1 score is: {f1}")
    #print("-------------------------------------")
    #print(f"Recall score is: {recall}")
    #print("-------------------------------------")
    #print(conf_matrix)
    print("********************************************************************")
    print("                           ***                                      ")



    ########################### Error analysis: select random samples #####################################
 
    # dataframe with only the wrong predictions
    df_wrong_predictions = df_output_prediction[df_output_prediction['prediction'] != df_output_prediction['gold_value']]
    print(df_wrong_predictions)
    # shuffle the dataframe
    df_wrong_predictions = df_wrong_predictions.sample(frac=1).reset_index(drop=True)
    # sample
    df_wrong_predictions = df_wrong_predictions[:10]
    print(df_wrong_predictions)
    # concatenate df wrong prediction to df errors
    df_errors = pd.concat([df_errors, df_wrong_predictions], ignore_index=True)

    # check if the entity in the col prediction is in the list of gold entities 
    # if not, it is a wrong prediction
    df_errors["error"] = df_errors["prediction"].apply(lambda x: "4" if x not in list_entity_types else "-")


# save the df to csv 
output_file = f"error_analysis/llama2_error_analysis.csv"
df_errors.to_csv(output_file, sep=";", index=False)