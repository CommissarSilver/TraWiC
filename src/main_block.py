import os, logging, logging.config, yaml, torch, argparse, random, json
from tqdm import tqdm
from checker import Checker, CheckerBlock
from models import SantaCoder, SantaCoderBlock
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
)  # programming language
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/the_stack/python",
    help="path to the dataset",
)
args = parser.parse_args()

model = SantaCoderBlock()


def get_model_output_inspector(file_path):
    results = []
    file_checker = CheckerBlock(file_path)
    model_inputs = file_checker.prepare_inputs_for_prediction()

    for candidate_input in tqdm(model_inputs):
        model_output = model.predict(candidate_input["prefix"])
        candidate_input["model_output"] = model_output
        results.append(candidate_input)       
    with open(os.path.join(os.getcwd(), "results_block.jsonl"), "a") as f:
        json_results = json.dumps(results)
        f.write(json_results)
        f.write("\n")

if __name__ == "__main__":
    dataset_files_path = [
        os.path.join(os.getcwd(), args.dataset_path, file)
        for file in os.listdir(os.path.join(os.getcwd(), args.dataset_path))
        if file.endswith(args.language)
    ]
    for file_path in dataset_files_path:
        results = []
        print("\033[91m" + file_path.split("/")[-1] + "\033[0m")
        result = get_model_output_inspector(file_path)
