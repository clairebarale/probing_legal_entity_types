import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, f1_score,recall_score
import argparse

#######################
# MLM cloze predictions
asylex_roberta = {
    "GPE": "mlm_cloze_text_entities_experiment/roberta-base/output_GPE_prompts_masked_with_context_template1.csv",
    "DATE": "mlm_cloze_text_entities_experiment/roberta-base/output_DATE_prompts_masked_with_context_template1.csv",
    "NORP": "mlm_cloze_text_entities_experiment/roberta-base/output_NORP_prompts_masked_with_context_template1.csv",
    "ORG": "mlm_cloze_text_entities_experiment/roberta-base/output_ORG_prompts_masked_with_context_template1.csv",
    "LAW": "mlm_cloze_text_entities_experiment/roberta-base/output_LAW_prompts_masked_with_context_template1.csv",
    "CREDIBILITY": "mlm_cloze_text_entities_experiment/roberta-base/output_CREDIBILITY_prompts_masked_with_context_template1.csv",
    "DETERMINATION": "mlm_cloze_text_entities_experiment/roberta-base/output_DETERMINATION_prompts_masked_with_context_template1.csv" ,
    "CLAIMANT_INFO": "mlm_cloze_text_entities_experiment/roberta-base/output_CLAIMANT_INFO_prompts_masked_with_context_template1.csv",
    "PROCEDURE": "mlm_cloze_text_entities_experiment/roberta-base/output_PROCEDURE_prompts_masked_with_context_template1.csv",
    "DOC_EVIDENCE": "mlm_cloze_text_entities_experiment/roberta-base/output_DOC_EVIDENCE_prompts_masked_with_context_template1.csv",
    "EXPLANATION": "mlm_cloze_text_entities_experiment/roberta-base/output_EXPLANATION_prompts_masked_with_context_template1.csv",
    "LEGAL_GROUND": "mlm_cloze_text_entities_experiment/roberta-base/output_LEGAL_GROUND_prompts_masked_with_context_template1.csv",
    "LAW_CASE": "mlm_cloze_text_entities_experiment/roberta-base/output_LAW_CASE_prompts_masked_with_context_template1.csv",
    "LAW_REPORT": "mlm_cloze_text_entities_experiment/roberta-base/output_LAW_REPORT_prompts_masked_with_context_template1.csv"
}

asylex_casehold = {
    "GPE": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_GPE_prompts_masked_with_context_template1.csv",
    "DATE": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_DATE_prompts_masked_with_context_template1.csv",
    "NORP": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_NORP_prompts_masked_with_context_template1.csv",
    "ORG": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_ORG_prompts_masked_with_context_template1.csv",
    "LAW": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_LAW_prompts_masked_with_context_template1.csv",
    "CREDIBILITY": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_CREDIBILITY_prompts_masked_with_context_template1.csv",
    "DETERMINATION": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_DETERMINATION_prompts_masked_with_context_template1.csv" ,
    "CLAIMANT_INFO": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_CLAIMANT_INFO_prompts_masked_with_context_template1.csv",
    "PROCEDURE": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_PROCEDURE_prompts_masked_with_context_template1.csv",
    "DOC_EVIDENCE": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_DOC_EVIDENCE_prompts_masked_with_context_template1.csv",
    "EXPLANATION": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_EXPLANATION_prompts_masked_with_context_template1.csv",
    "LEGAL_GROUND": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_LEGAL_GROUND_prompts_masked_with_context_template1.csv",
    "LAW_CASE": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_LAW_CASE_prompts_masked_with_context_template1.csv",
    "LAW_REPORT": "mlm_cloze_text_entities_experiment/casehold/legalbert/output_LAW_REPORT_prompts_masked_with_context_template1.csv"
}

asylex_lexlm = {
    "GPE": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_GPE_prompts_masked_with_context_template1.csv",
    "DATE": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_DATE_prompts_masked_with_context_template1.csv",
    "NORP": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_NORP_prompts_masked_with_context_template1.csv",
    "ORG": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_ORG_prompts_masked_with_context_template1.csv",
    "LAW": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_LAW_prompts_masked_with_context_template1.csv",
    "CREDIBILITY": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_CREDIBILITY_prompts_masked_with_context_template1.csv",
    "DETERMINATION": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_DETERMINATION_prompts_masked_with_context_template1.csv" ,
    "CLAIMANT_INFO": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_CLAIMANT_INFO_prompts_masked_with_context_template1.csv",
    "PROCEDURE": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_PROCEDURE_prompts_masked_with_context_template1.csv",
    "DOC_EVIDENCE": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_DOC_EVIDENCE_prompts_masked_with_context_template1.csv",
    "EXPLANATION": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_EXPLANATION_prompts_masked_with_context_template1.csv",
    "LEGAL_GROUND": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_LEGAL_GROUND_prompts_masked_with_context_template1.csv",
    "LAW_CASE": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_LAW_CASE_prompts_masked_with_context_template1.csv",
    "LAW_REPORT": "mlm_cloze_text_entities_experiment/lexlms/legal-roberta-large/output_LAW_REPORT_prompts_masked_with_context_template1.csv"
}

asylex_deberta = {
    "GPE": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_GPE_prompts_masked_with_context_template1.csv",
    "DATE": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_DATE_prompts_masked_with_context_template1.csv",
    "NORP": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_NORP_prompts_masked_with_context_template1.csv",
    "ORG": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_ORG_prompts_masked_with_context_template1.csv",
    "LAW": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_LAW_prompts_masked_with_context_template1.csv",
    "CREDIBILITY": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_CREDIBILITY_prompts_masked_with_context_template1.csv",
    "DETERMINATION": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_DETERMINATION_prompts_masked_with_context_template1.csv" ,
    "CLAIMANT_INFO": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_CLAIMANT_INFO_prompts_masked_with_context_template1.csv",
    "PROCEDURE": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_PROCEDURE_prompts_masked_with_context_template1.csv",
    "DOC_EVIDENCE": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_DOC_EVIDENCE_prompts_masked_with_context_template1.csv",
    "EXPLANATION": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_EXPLANATION_prompts_masked_with_context_template1.csv",
    "LEGAL_GROUND": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_LEGAL_GROUND_prompts_masked_with_context_template1.csv",
    "LAW_CASE": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_LAW_CASE_prompts_masked_with_context_template1.csv",
    "LAW_REPORT": "mlm_cloze_text_entities_experiment/microsoft/deberta-v3-base/output_LAW_REPORT_prompts_masked_with_context_template1.csv"
}

asylex_pol = {
    "GPE": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_GPE_prompts_masked_with_context_template1.csv",
    "DATE": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_DATE_prompts_masked_with_context_template1.csv",
    "NORP": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_NORP_prompts_masked_with_context_template1.csv",
    "ORG": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_ORG_prompts_masked_with_context_template1.csv",
    "LAW": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_LAW_prompts_masked_with_context_template1.csv",
    "CREDIBILITY": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_CREDIBILITY_prompts_masked_with_context_template1.csv",
    "DETERMINATION": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_DETERMINATION_prompts_masked_with_context_template1.csv" ,
    "CLAIMANT_INFO": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_CLAIMANT_INFO_prompts_masked_with_context_template1.csv",
    "PROCEDURE": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_PROCEDURE_prompts_masked_with_context_template1.csv",
    "DOC_EVIDENCE": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_DOC_EVIDENCE_prompts_masked_with_context_template1.csv",
    "EXPLANATION": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_EXPLANATION_prompts_masked_with_context_template1.csv",
    "LEGAL_GROUND": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_LEGAL_GROUND_prompts_masked_with_context_template1.csv",
    "LAW_CASE": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_LAW_CASE_prompts_masked_with_context_template1.csv",
    "LAW_REPORT": "mlm_cloze_text_entities_experiment/pile-of-law/legalbert-large-1.7M-2/output_LAW_REPORT_prompts_masked_with_context_template1.csv"
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


# convert to list
list_entity_types = list(dict_entity_types.values())
# flatten the list
list_entity_types = [item for sublist in list_entity_types for item in sublist]
print(list_entity_types)

#######################
print("*** AsyLex dataset ***")

# define the model to use here: 
asylex_model = asylex_lexlm

# create dataframe to store all erros per model
df_errors = pd.DataFrame()

# get the file with prediction adn the list of gold entities per entity type
for ent_type in asylex_model:
    print(f"***Entity: {ent_type}***")

    df_output_prediction = pd.read_csv(asylex_model[ent_type], sep=';', usecols=['entity', 'prompt', 'output_prediction'])
    list_gold_entities = dict_entity_types[ent_type]
    # replace values in the dataframe that are in list by entity type
    # this way whatever the answer, as long as it is a entity type, it will be counted as correct
    df_output_prediction = df_output_prediction.replace(list_gold_entities, ent_type)
    df_output_prediction["gold_value"] = ent_type

    # count number of occurences of DATE in the dataframe
    count = df_output_prediction["output_prediction"].value_counts()[ent_type]
    print(f"Number of occurences of {ent_type}: ", count)

    df_output_prediction.astype(str, errors='ignore')
    df_output_prediction = df_output_prediction.dropna()
    print(len(df_output_prediction))
    
    y_true = df_output_prediction["gold_value"]
    y_pred = df_output_prediction['output_prediction']
    print(df_output_prediction)


    # Generate the classification report
    report = classification_report(y_true, y_pred, digits=4)
    f1 = f1_score(y_true, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_true, y_pred,)
    recall = recall_score(y_true, y_pred, average="weighted")
    print(report)
    print("-------------------------------------")
    print(f"F1 score is: {f1}")
    print("-------------------------------------")
    print(f"Recall score is: {recall}")
    print("-------------------------------------")
    print(conf_matrix)
    print("********************************************************************")
    print("                           ***                                      ")

########################### Error analysis: samples #####################################
    
 
    # dataframe with only the wrong predictions
    df_wrong_predictions = df_output_prediction[df_output_prediction['output_prediction'] != df_output_prediction['gold_value']]
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
    df_errors["error"] = df_errors["output_prediction"].apply(lambda x: "4" if x not in list_entity_types else "-")


# save the df to csv    
output_file = f"error_analysis/lexlm_error_analysis.csv"
df_errors.to_csv(output_file, sep=";", index=False)
    
