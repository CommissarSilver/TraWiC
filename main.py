import os, logging
from utils import dataset, process_scripts

logging.basicConfig(
    filename="runlog.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    dataset.get_thestack_dataset(
        language="python",
        save_directory=os.path.join(os.getcwd(), "data"),
        scripts_num=10**4,
    )
    process_scripts.word_count_directory(
        directory_path=os.path.join(os.getcwd(), "data", "the_stack", "python"),
        script_suffix=".py",
    )
