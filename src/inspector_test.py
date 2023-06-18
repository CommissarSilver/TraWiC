import os
import multiprocessing
from tqdm import tqdm
from data import comment_to_code_ratio


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
    ignored_files = trained_files + test_files + valid_files
    all_files = [
        i
        for i in os.listdir(os.path.join(os.getcwd(), "data", "the_stack", "python"))
        if i not in ignored_files
    ]

    with multiprocessing.Pool() as pool:
        files_dict = dict(tqdm(pool.imap(process_file, all_files), total=len(all_files)))

    with open(os.path.join(os.getcwd(), "ds_python.csv"), "w") as f:
        f.write("file_name,label\n")
        for file_name, label in files_dict.items():
            f.write(f"{file_name},{label}\n")
