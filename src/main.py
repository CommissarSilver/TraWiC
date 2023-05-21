import os, logging, logging.config, yaml

from data import get_thestack_dataset
from utils import word_count_directory
from models import SantaCoder
from checker import Checker

# load logging configuration
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

if __name__ == "__main__":
    # get_thestack_dataset(
    #     language="python",
    #     save_directory=os.path.join(os.getcwd(), "data"),
    #     scripts_num=100,
    # )
    # word_count_directory(
    #     directory_path=os.path.join(os.getcwd(), "data", "the_stack", "python"),
    #     script_suffix=".py",
    # )

    test_input = Checker(
        "/Users/ahura/Nexus/TWMC/data/the_stack/python/the_stack_python_script_0.py"
    )

    candidate_function_inputs = test_input.prepare_inputs_for_infill(
        level="function_names"
    )
    candidate_variable_inputs = test_input.prepare_inputs_for_infill(
        level="variable_names"
    )
    candidate_class_inputs = test_input.prepare_inputs_for_infill(level="class_names")
    candidate_comment_inputs = test_input.prepare_inputs_for_infill(level="comments")
    candidate_docstring_inputs = test_input.prepare_inputs_for_infill(level="docstrings")
    candidate_string_inputs = test_input.prepare_inputs_for_infill(level="strings")

    candidate_inputs = {
        "function_names": {
            "inputs": candidate_function_inputs,
            "similarity_metric": "exact",
        },
        "variable_names": {
            "inputs": candidate_variable_inputs,
            "similarity_metric": "exact",
        },
        "class_names": {
            "inputs": candidate_class_inputs,
            "similarity_metric": "exact",
        },
        "comments": {
            "inputs": candidate_comment_inputs,
            "similarity_metric": "fuzzy",
        },
        "docstrings": {
            "inputs": candidate_docstring_inputs,
            "similarity_metric": "fuzzy",
        },
        "strings": {
            "inputs": candidate_string_inputs,
            "similarity_metric": "fuzzy",
        },
    }
    model = SantaCoder()

    results = []
    for candidate_input_level, candidate_input in candidate_inputs.items():
        inputs = candidate_input["inputs"]

        model_output = model.infill((inputs["prefix"], inputs["suffix"]))

        result = test_input.check_similarity(
            model_output,
            candidate_input,
            similiarity_metric=candidate_input["similarity_metric"],
        )
        results.append(result)
    # save the results to a json file
    import json

    json.dump(results, open(os.path.join(os.getcwd(), "results.json"), "w"))
    # if False:
    #     if candidate_input["infill"] in model_output:
    #         print(
    #             "\033[92m"
    #             + candidate_input["prefix"]
    #             + "\033[93m"
    #             + model_output
    #             + "\033[92m"
    #             + candidate_input["suffix"]
    #         )
    #         print(
    #             model_output.count(
    #                 "\n", 0, model_output.find(candidate_input["infill"])
    #             )
    #         )

    # j = model.infill((candidate_inputs[0]["prefix"], candidate_inputs[0]["suffix"]))
    # search for the infill in the original script
    # if candidate_inputs[0]["infill"] in j:
    # print("found it")
    # print("\033[92m" + candidate_inputs[0]["prefix"] + "\033[93m" + j + "\033[92m" + candidate_inputs[0]["suffix"])
    print("hi")
