import os
import json
import random
import shutil
from tqdm import tqdm

NICAD_DIR = os.path.join("/", "Users", "ahura", "Downloads", "NiCad-6.2")
WORKING_DIR = os.path.join(os.getcwd())


def copy_python_files(src, dest):
    for dirpath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            if filename.endswith(".py"):
                source_item = os.path.join(dirpath, filename)
                target_item = os.path.join(dest, src.split("/")[-1] + "_" + filename)
                shutil.copy(source_item, target_item)


def process_directory(directory, selected_directories):
    os.chdir(WORKING_DIR)
    source = os.path.join(os.getcwd(), "src", "blocks", directory)
    target = os.path.join(NICAD_DIR, "systems", "analysis_target")

    random_directory_paths = [
        os.path.join(os.getcwd(), "src", "blocks", directory)
        for directory in selected_directories
    ]

    print(
        "\033[92m"
        + "Moving from TWMC to NICAD ->"
        + os.path.join(NICAD_DIR, "systems", directory)
        + "\033[0m"
    )
    os.makedirs(os.path.join(NICAD_DIR, "systems", "analysis_target"), exist_ok=True)
    for path in [source] + random_directory_paths:
        copy_python_files(path, target)

    print("\033[93m" + f"Running NiCAD Block on {directory}" + "\033[0m")
    os.chdir(NICAD_DIR)
    os.system(
        f"arch -x86_64 sh {NICAD_DIR}/nicad6 blocks py {NICAD_DIR}/systems/analysis_target"
    )

    html_files = [
        file
        for file in os.listdir(
            os.path.join(
                NICAD_DIR,
                "systems",
                f"analysis_target_blocks-blind-clones",
            )
        )
        if file.endswith(".html")
    ]
    with open(
        os.path.join(
            NICAD_DIR,
            "systems",
            "analysis_target_blocks-blind-clones",
            html_files[0],
        )
    ) as f:
        nicad_results[directory] = f.read()

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
    systems_directory = os.path.join(NICAD_DIR, "systems")
    for item in os.listdir(systems_directory):
        item_path = os.path.join(systems_directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


if __name__ == "__main__":
    NUM_SAMPLES = 20
    directories = os.listdir(os.path.join(os.getcwd(), "src", "blocks"))
    for directory in directories:
        selected_directories = random.sample(
            directories, min(NUM_SAMPLES, len(directories))
        )
        nicad_results = {}
        process_directory(directory, selected_directories)
