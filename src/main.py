import os
from data import dataset
from utils import process_scripts
from utils.santa import SantaCoder


if __name__ == "__main__":
    model = SantaCoder()
    print(model.predict("import numpy as np"))
    dataset.get_thestack_dataset(
        language="python",
        save_directory=os.path.join(os.getcwd(), "data"),
        scripts_num=10**4,
    )
    process_scripts.word_count_directory(
        directory_path=os.path.join(os.getcwd(), "data", "the_stack", "python"),
        script_suffix=".py",
    )
