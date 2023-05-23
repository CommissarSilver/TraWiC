import os, logging, logging.config, yaml, torch, argparse, random, json
from checker import Checker
from models import SantaCoder


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

    model_inputs = [input for sublist in model_inputs for input in sublist]
    results = []
    for candidate_input in model_inputs[:2]:
        model_output = model.infill(
            (candidate_input["prefix"], candidate_input["suffix"])
        )

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
                "similarity_metric": "exact"
                if candidate_input["level"]
                in ["function_names", "variable_names", "class_names"]
                else "fuzzy",
                "result": result,
            }
        )

    json.dump(results, open(os.path.join(os.getcwd(), "results.json"), "w"))
