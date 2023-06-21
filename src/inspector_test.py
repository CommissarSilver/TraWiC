import os
import multiprocessing
from tqdm import tqdm
from data import comment_to_code_ratio
import pandas as pd

trained_files = []
test_files = []
valid_files = []


def process_file(file_name):
    try:
        return (
            file_name,
            1
            if 0.01
            < comment_to_code_ratio(
                os.path.join(os.getcwd(), "data", "the_stack", "python", file_name)
            )
            < 0.8
            else 0,
        )
    except:
        return (file_name, 2)


if __name__ == "__main__":
    # ignored_files = trained_files + test_files + valid_files
    # all_files = [
    #     i
    #     for i in os.listdir(os.path.join(os.getcwd(), "data", "the_stack", "python"))
    #     if i not in ignored_files
    # ]

    # with multiprocessing.Pool() as pool:
    #     files_dict = dict(tqdm(pool.imap(process_file, all_files), total=len(all_files)))

    # with open(os.path.join(os.getcwd(), "ds_python.csv"), "w") as f:
    #     f.write("file_name,label\n")
    #     for file_name, label in files_dict.items():
    #         f.write(f"{file_name},{label}\n")
    trained_ons = pd.read_csv("/Users/ahura/Nexus/TWMC/src/trained_on.csv")[
        "file_name"
    ].tolist()
    entire_ds = pd.read_csv("/Users/ahura/Nexus/TWMC/ds_python.csv")
    # drop the files that are already in the trained_on.csv
    entire_ds = entire_ds[~entire_ds["file_name"].isin(trained_ons)]
    # filter by the ones who have a 0 label
    entire_ds = entire_ds[entire_ds["label"] == 0]
    entire_ds.to_csv("/Users/ahura/Nexus/TWMC/src/ds_python_neg.csv", index=False)
