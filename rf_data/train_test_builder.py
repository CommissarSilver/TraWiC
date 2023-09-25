import os
import tqdm
import pandas as pd

# only get the folders
# all_directories=[os.path.join(os.getcwd(),'rf_data',path)for path in os.listdir(os.path.join(os.getcwd(),'rf_data')) if os.path.isdir(os.path.join(os.getcwd(),'rf_data',path))]
all_directories=['/store/travail/vamaj/TWMC/rf_data/syn100_sem10']

# Read all the csv files in the directory, and concatenate them into a single dataframe
for directory in tqdm.tqdm(all_directories):
    df = pd.concat([pd.read_csv(os.path.join(directory,file)) for file in os.listdir(directory) if file.endswith('.csv')])
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    # drop useless columns
    df.drop(columns=['class_nums_total','function_nums_total','variable_nums_total','string_nums_total','comment_nums_total','docstring_nums_total'],inplace=True)
    # save the concatenated dataframe as a csv file
    df.to_csv(os.path.join(directory,'train.csv'),index=False)
    

