import os, logging, logging.config, yaml

from data import dataset
from utils import process_scripts
from models.santa import SantaCoder
from checker.checker import Checker

# load logging configuration
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

if __name__ == "__main__":
    model = SantaCoder()

    # dataset.get_thestack_dataset(
    #     language="python",
    #     save_directory=os.path.join(os.getcwd(), "data"),
    #     scripts_num=10,
    # )
    # process_scripts.word_count_directory(
    #     directory_path=os.path.join(os.getcwd(), "data", "the_stack", "python"),
    #     script_suffix=".py",
    # )
    test_input = Checker(
        "/Users/ahura/Nexus/TWMC/data/the_stack/python/the_stack_python_script_0.py"
    )

    candidate_inputs = test_input.prepare_inputs_for_infill(level="function_names")
    for candidate_input in candidate_inputs:
        model_output = model.infill(
            (candidate_input["prefix"], candidate_input["suffix"])
        )
        test_input.check_similarity(
            model_output, candidate_input, similiarity_metric="exact"
        )
        if False:
            if candidate_input["infill"] in model_output:
                print(
                    "\033[92m"
                    + candidate_input["prefix"]
                    + "\033[93m"
                    + model_output
                    + "\033[92m"
                    + candidate_input["suffix"]
                )
                print(
                    model_output.count(
                        "\n", 0, model_output.find(candidate_input["infill"])
                    )
                )

    # j = model.infill((candidate_inputs[0]["prefix"], candidate_inputs[0]["suffix"]))
    # search for the infill in the original script
    # if candidate_inputs[0]["infill"] in j:
    # print("found it")
    # print("\033[92m" + candidate_inputs[0]["prefix"] + "\033[93m" + j + "\033[92m" + candidate_inputs[0]["suffix"])
    print("hi")
