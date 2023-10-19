import pandas as pd
import glob



source_folder = "error_analysis/"
# loop through the files in the folder and open them
file_names = f"{source_folder}/llama2_error_analysis.csv"

for filepath in glob.iglob(file_names, recursive=True):
    print("-------------------------------------")
    print(filepath)
    df = pd.read_csv(filepath, sep=";")
    print(len(df))
    # count the number of occurences of each error type
    print(df["error"].value_counts())


        
