import os, logging, logging.config, yaml, torch, argparse, random, json
from tqdm import tqdm
from checker import Checker
from models import SantaCoder
import pandas as pd

# load logging configuration
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(description="Trained Without My Consent")
parser.add_argument(
    "--language",
    type=str,
    default="py",
    help="language of the code",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/the_stack/python",
    help="path to the dataset",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size",
)
args = parser.parse_args()

model = SantaCoder()

dataset_files_path = [
    os.path.join(os.getcwd(), args.dataset_path, file)
    for file in os.listdir(os.path.join(os.getcwd(), args.dataset_path))
    if file.endswith(args.language)
]
results_so_far = pd.read_csv(os.path.join(os.getcwd(), "src", "trained_on.csv"))
resluts_so_far_names = results_so_far["file_name"].tolist()

dataset_files_path = [
    file for file in dataset_files_path if file.split("/")[-1] not in resluts_so_far_names
]


for file_path in dataset_files_path:
    # select randomly from dataset_files_path
    file_checker = Checker(file_path)

    model_inputs = [
        file_checker.prepare_inputs_for_infill(level=i)
        for i in [
            "function_names",
            "variable_names",
            "class_names",
            "comments",
            "docstrings",
            "strings",
        ]
    ]
    # print file path in red
    print("\033[91m" + file_path + "\033[0m")
    model_inputs = [input for sublist in model_inputs for input in sublist]
    if model_inputs == []:
        continue
    for candidate_input in tqdm(model_inputs):
        results = []

        model_output = model.infill(
            (candidate_input["prefix"], candidate_input["suffix"])
        )
        if model_output == "too_many_tokens":
            print("\033[91m" + file_path + "has too many tokens - skipping" + "\033[0m")
            f = open(os.path.join(os.getcwd(), "too_many_tokens.txt"), "a")
            f.write(file_path + "\n")
            break
        try:
            result = file_checker.check_similarity(
                model_output,
                candidate_input,
                similiarity_metric="exact"
                if candidate_input["level"]
                in ["function_names", "variable_names", "class_names"]
                else "fuzzy",
            )
            results.append(
                {
                    "file_path": file_path,
                    "level": candidate_input["level"],
                    "similarity_metric": "exact"
                    if candidate_input["level"]
                    in ["function_names", "variable_names", "class_names"]
                    else "fuzzy",
                    "result": result,
                    "similarity_objective": candidate_input["infill"],
                    "model_output": model_output,
                }
            )
        except Exception as e:
            logging.error(e)

        with open(os.path.join(os.getcwd(), "results.jsonl"), "a") as f:
            for result in results:
                json_results = json.dumps(results)
                f.write(json_results)
                f.write("\n")
    # add to results.json don't overwrite
