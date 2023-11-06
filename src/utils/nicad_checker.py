import json
import multiprocessing as mp
import os
import random
import re
import shutil
from typing import Dict, List

from bs4 import BeautifulSoup
from tqdm import tqdm

NICAD_DIR = os.path.join("/", "store", "travail", "vamaj", "TXL2", "NiCad-6.2")
WORKING_DIR = os.path.join(os.getcwd())


def copy_python_files(src: str, dest: str) -> None:
    """
    Copies all Python files from src to dest.

    Args:
        src (str): Source directory
        dest (str): Destination directory
    """

    for dirpath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            if filename.endswith(".py"):
                source_item = os.path.join(dirpath, filename)
                target_item = os.path.join(dest, src.split("/")[-1] + "_" + filename)
                shutil.copy(source_item, target_item)


def process_directory(
    directory: str, selected_directories: list, core_number: int
) -> None:
    """
    Processes the files in a directory by running the NiCAD colne detector on them.

    1 - Copies the files from the directory and the randomly selected directories to the NiCAD directory
    2 - Runs NiCAD on the files
    3 - Saves the results in a json file
    4 - Removes the files from the NiCAD directory

    Args:
        directory (str): the main directory to run clone detection on
        selected_directories (list): randomly selected directories to run clone detection against
    """

    # Skip processing if the results already exist
    if os.path.exists(
        os.path.join(
            WORKING_DIR,
            "nicad_results",
            "original",
            f"nicad_results_{directory}.json",
        )
    ):
        return

    # change the working directory to the NiCAD directory
    os.chdir(WORKING_DIR)
    source = os.path.join(os.getcwd(), "blocks", directory)
    target = os.path.join(NICAD_DIR, "systems", f"analysis_target_{core_number}")

    # get the full paths of the randomly selected directories
    random_directory_paths = [
        os.path.join(os.getcwd(), "blocks", directory)
        for directory in selected_directories
    ]
    # align the prints so that they are easier to read
    print(
        "\033[92m"
        + f"{core_number}".ljust(2)
        + " - "
        + "Moving from TWMC to NICAD -> ".ljust(30)
        + os.path.join(NICAD_DIR, "systems", directory).ljust(50)
        + "\033[0m"
    )
    # make a directory named analysis_target which will contain all the files to be analyzed
    os.makedirs(
        os.path.join(NICAD_DIR, "systems", f"analysis_target_{core_number}"),
        exist_ok=True,
    )
    for path in [source] + random_directory_paths:
        copy_python_files(path, target)

    # change the working directory to the NiCAD directory and run NiCAD
    print(
        "\033[93m"
        + f"{core_number}".ljust(2)
        + " - "
        + f"Running NiCAD Block on {directory}".ljust(50)
        + "\033[0m"
    )
    os.chdir(NICAD_DIR)
    # no need to display terminal output
    os.system(
        f"sh {NICAD_DIR}/nicad6 blocks py {NICAD_DIR}/systems/analysis_target_{core_number} > /dev/null 2>&1",
    )
    # NiCAD produces results as an html file, so we read the html file and save it as a json file. The reason for storing them and not processing them outright is to have a record of the results in case something goes wrong.
    html_files = [
        file
        for file in os.listdir(
            os.path.join(
                NICAD_DIR,
                "systems",
                f"analysis_target_{core_number}_blocks-blind-clones",
            )
        )
        if file.endswith(".html")
    ]
    nicad_results = {}
    # read the original html file, process it and save it as a json file
    with open(
        os.path.join(
            NICAD_DIR,
            "systems",
            f"analysis_target_{core_number}_blocks-blind-clones",
            html_files[0],
        )
    ) as f:
        nicad_results[directory] = f.read()
        result_dict = parse_clone_classes_and_files(nicad_results[directory])

        results_directory_path = "/store/travail/vamaj/TWMC/nicad_results/results"
        json_file_path = os.path.join(results_directory_path, f"{directory}_result.json")

        with open(json_file_path, "w") as json_file:
            json.dump(result_dict, json_file)

        print(
            "\033[93m"
            + f"{core_number}".ljust(2)
            + " - "
            + f"Read HTML files on {directory}".ljust(50)
            + "\033[0m"
        )

    json.dump(
        nicad_results,
        open(
            os.path.join(
                WORKING_DIR,
                "nicad_results",
                "original",
                f"nicad_results_{directory}.json",
            ),
            "w",
        ),
    )
    # remove the files from the NiCAD directory for the next iteration
    systems_directory = os.path.join(NICAD_DIR, "systems")
    pattern = f"analysis_target_{core_number}"
    for item in os.listdir(systems_directory):
        if re.match(pattern, item):
            item_path = os.path.join(systems_directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

            print(
                "\033[93m"
                + f"{core_number}".ljust(2)
                + " - "
                + f"Removed files on {directory}-{item}".ljust(50)
                + "\033[0m"
            )


def parse_clone_classes_and_files(html_content: str) -> Dict[str, List[str]]:
    """
    Parses the given HTML content to extract clone classes and file names.

    Args:
        html_content: A string containing the HTML content.
    Returns:
        A dictionary containing the extracted clone classes and file names.
        The keys are in the format 'clone_class_X', where X is the index of the clone class,
        and the values are lists of file names within that clone class.
    """
    # Initialize the dictionary to store the results
    clone_classes_dict: Dict[str, List[str]] = {}

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all tables that contain clone classes and file names
    tables = soup.find_all("table", border="2", cellpadding="4")

    # Iterate through the tables and extract the information
    for clone_class_index, table in enumerate(tables, start=1):
        # Extract all file names within this clone class
        file_links = table.find_all("a")
        file_names = [link.text.strip() for link in file_links]

        # Add to the dictionary using the clone class index as the key
        clone_classes_dict[f"clone_class_{clone_class_index}"] = file_names

    return clone_classes_dict


def check_repo(repo_name: str, clone_classes: dict) -> bool:
    """
    Recieves the repository name alongside all the detected cloning results.
    If even one file in the repo is detected to be clone of another file from another repo, the function returns True.

    Args:
        repo_name (str): name of the repository
        clone_classes (dict): dictionary of clone classes and their files

    Returns:
        bool: whether the repo is a clone or not
    """
    # create a dictionary of the repo name and a list of booleans indicating whether the repo name is in the file name or not
    repo_masks: dict[str, list[bool]] = {
        key: [repo_name in string for string in string_list]
        for key, string_list in clone_classes.items()
    }
    # create a dictionary of the clone classes that contain the repo name
    remain_repo: dict[str, list[bool]] = {
        key: value for key, value in clone_classes.items() if True in repo_masks[key]
    }
    # count the number of times the repo name appears in the original and generated files
    results_count: dict[str, dict[str, int]] = {
        key: {
            "original": sum("original" in string for string in string_list),
            "generated": sum("generated" in string for string in string_list),
        }
        for key, string_list in remain_repo.items()
    }
    # return true if even one file is detected to be a clone of another file from another repo
    results: bool = (
        True
        if any(
            [
                value["original"] * value["generated"] > 0
                for value in results_count.values()
            ]
        )
        else False
    )

    return results


def worker_function(args):
    # number of randomly selected directories to run clone detection against
    NUM_SAMPLES = int(0.1 * len(os.listdir(os.path.join(os.getcwd(), "blocks"))))
    print(f"Number of samples: {NUM_SAMPLES}")
    # 0.05 equals to 68 repos
    # 0.09 equals to 122 repos
    # 0.1 equals to 136 repos
    directories_chunk, cpu_core_number = args

    directories = os.listdir(os.path.join(os.getcwd(), "blocks"))

    for directory in directories_chunk:
        selected_directories = random.sample(directories, NUM_SAMPLES)

        process_directory(directory, selected_directories, cpu_core_number)


if __name__ == "__main__":
    num_cores = mp.cpu_count()
    directories = os.listdir(os.path.join(os.getcwd(), "blocks"))

    # divide the directories into chunks and assign each chunk to a core
    chunk_size = len(directories) // num_cores
    directories_chunks = [
        directories[i : i + chunk_size] for i in range(0, len(directories), chunk_size)
    ]
    directories_with_core_numbers = [
        (chunk, i) for i, chunk in enumerate(directories_chunks)
    ]
    # process_directory(directories[0], directories[1:3], 0)
    with mp.Pool(num_cores) as pool:
        #! Number of repo samples is set inside the worker function
        pool.map(worker_function, directories_with_core_numbers)
        pool.close()
        pool.join()

    # Directory containing the original JSON files
    original_directory_path = os.path.join(
        "/store", "travail", "vamaj", "TWMC", "nicad_results", "original"
    )
    # Directory to save the result JSON files
    results_directory_path = os.path.join(
        "/store", "travail", "vamaj", "TWMC", "nicad_results", "results"
    )

    # Create the results directory if it doesn't exist
    os.makedirs(results_directory_path, exist_ok=True)
    # Iterate over all the files in the original directory
    for filename in os.listdir(original_directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(original_directory_path, filename), "r") as f:
                nicad_results = json.load(f)
                html_content = list(nicad_results.values())[0]

            # Parse the HTML content and get the result
            result_dict = parse_clone_classes_and_files(html_content)
            repo_detected = check_repo(
                filename.strip("nicad_results_").strip(".json"), result_dict
            )
            # Save the result as a JSON file in the results directory
            json_file_path = os.path.join(
                results_directory_path, filename.replace(".json", "_result.json")
            )
            with open(json_file_path, "w") as json_file:
                json.dump(result_dict, json_file)

            if not os.path.exists(os.path.join(os.getcwd(), "NiCAD_results.csv")):
                with open(os.path.join(os.getcwd(), "NiCAD_results.csv"), "w") as f:
                    f.write("filename,repo_detected\n")

            with open(os.path.join(os.getcwd(), "NiCAD_results.csv"), "a") as f:
                f.write(
                    f"{filename.strip('nicad_results_').strip('.json')},{repo_detected}\n"
                )
