import json
import os

import pandas as pd


def parse_repo_and_trained_on(token_results: pd.DataFrame):
    # Initialize the dictionary to store the result
    repo_dict = {}

    # Iterate through the DataFrame to extract the repository name and check the "trained_on" column
    for index, row in token_results.iterrows():
        # Extract the repository name by splitting the string at the first underscore
        repo_name = row["Unnamed: 0"].split("/")[0]

        # Check if the repository name is already in the dictionary
        if repo_name not in repo_dict.keys():
            # If not, add the repository name to the dictionary with the value from "trained_on"
            repo_dict[repo_name] = row["trained_on"]

    return repo_dict


files = [
    pd.read_csv(file)
    for file in [
        "/Users/ahura/Nexus/TWMC/TokensRun0_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun1_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun2_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun3_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun4_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun5_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun6_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun7_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun8_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun9_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun10_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun11_processed_dataset.csv",
        "/Users/ahura/Nexus/TWMC/TokensRun12_processed_dataset.csv",
    ]
]
final_df = pd.concat(files)
parse_repo_and_trained_on(final_df)
# save it in a csv file
with open("/Users/ahura/Nexus/TWMC/repo_trained_on.csv", "a") as f:
    f.write("repo_name,trained_on\n")
    for key, value in parse_repo_and_trained_on(final_df).items():
        f.write(f"{key},{value}\n")
