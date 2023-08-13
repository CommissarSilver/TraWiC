import os
import json
import multiprocessing
import shutil
from tqdm import tqdm

NICAD_DIR = os.path.join("/", "Users", "ahura", "Downloads", "NiCad-6.2")
WORKING_DIR = os.path.join(os.getcwd())


def process_directory(directory):
    os.chdir(WORKING_DIR)
    source = os.path.join(os.getcwd(), "src", "blocks", directory)
    target = os.path.join(NICAD_DIR, "systems")

    print(
        "\033[92m"
        + "Moving from TWMC to NICAD ->"
        + os.path.join(NICAD_DIR, "systems", directory)
        + "\033[0m"
    )
    os.system(f"mkdir -p {target} && cp -r {source} {target}")

    print("\033[93m" + f"Running NiCAD Block on {directory}" + "\033[0m")
    os.chdir(NICAD_DIR)
    os.system(
        f"arch -x86_64 sh {NICAD_DIR}/nicad6 blocks py {NICAD_DIR}/systems/{directory}"
    )

    html_files = [
        file
        for file in os.listdir(
            os.path.join(
                NICAD_DIR,
                "systems",
                f"{directory}_blocks-blind-clones",
            )
        )
        if file.endswith(".html")
    ]
    with open(
        os.path.join(
            NICAD_DIR,
            "systems",
            f"{directory}_blocks-blind-clones",
            html_files[0],
        )
    ) as f:
        nicad_results[directory] = f.read()

    json.dump(
        nicad_results,
        open(os.path.join(WORKING_DIR, f"nicad_results_{directory}.json"), "w"),
    )
    systems_directory = os.path.join(NICAD_DIR, "systems")
    for item in os.listdir(systems_directory):
        item_path = os.path.join(systems_directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


if __name__ == "__main__":
    nicad_results = {}
    directories = os.listdir(os.path.join(os.getcwd(), "src", "blocks"))
    for directory in directories:
        process_directory(directory)
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     for _ in tqdm(
    #         pool.imap_unordered(process_directory, directories),
    #         total=len(directories),
    #         desc="Processing",
    #     ):
    #         if KeyboardInterrupt:
    #             pool.terminate()
    #             pool.join()
    #             break
