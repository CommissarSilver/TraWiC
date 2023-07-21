import os, logging, logging.config, yaml, torch, argparse, random, json, warnings
from tqdm import tqdm
from checker import Checker, CheckerBlock
from models import SantaCoder, SantaCoderBlock
import pandas as pd


# Disable all warnings
warnings.filterwarnings("ignore")
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
    default="data",
    help="path to the dataset",
)
args = parser.parse_args()

model = SantaCoderBlock()
# if gpu is available, use it and send it to the gpu device
if torch.cuda.is_available():
    model=model.cuda()

def get_model_output_inspector(file_path):
    results = []
    file_checker = CheckerBlock(file_path)
    model_inputs = file_checker.prepare_inputs_for_prediction()

    for candidate_input in tqdm(model_inputs):
        model_output = model.predict(candidate_input["prefix"], candidate_input["suffix"])
        candidate_input["model_output"] = model_output
        results.append(candidate_input)
    with open(os.path.join(os.getcwd(), "results_block.jsonl"), "a") as f:
        json_results = json.dumps(results)
        f.write(json_results)
        f.write("\n")


if __name__ == "__main__":
    # print all gpu devices available
    print("Available devices: ", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("""⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠖⠒⠒⠒⠒⠦⢤⣀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣦⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠁⠀⠀⣾⣛⣛⣖⣒⠦⣀⠀⠀⠀⢻⢳⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠀⠀⢀⣀⣉⠀⠀⠀⠈⠓⢎⢷⡀⠀⠈⡆⢷⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠃⠀⡾⠛⣄⠈⠙⣆⣠⣤⣤⣈⠙⠁⠀⠀⡇⠘⡆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠸⣇⠀⠉⠀⢀⡟⠁⢀⡀⠙⣿⠀⠀⠀⡇⠀⣷
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⠀⣀⣀⣿⣲⠤⡴⠛⣇⠀⠈⠉⢁⡿⠀⠀⣸⠁⠀⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠁⡞⡟⢻⣿⣄⣹⣡⠖⠈⣷⣲⣖⡏⠀⠀⠀⡏⠀⠀⡿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡇⠀⢇⢧⠀⠘⠻⣽⣻⣶⣶⣶⠯⣟⡄⠀⠀⣸⠀⠀⢰⠃
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡏⠀⠀⠈⠫⡿⣶⣄⡀⢰⣟⡛⢦⡀⢸⢸⠀⠀⡇⠀⠀⡼⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⢶⡏⠁⠀⠀⠈⠈⠙⠚⠽⣶⣿⣾⣿⡻⠏⠀⢸⠀⠀⢠⠇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠃⠈⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⢠⠉⠉⠁⠀⢀⡇⠀⠀⣞⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⣼⠀⠀⢀⡽⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠸⠁⠀⠀⡿⠁⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣞⣠⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠀⡚⡷⠀⠀⣼⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⡿⢹⣷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇⠀⠘⠛⠃⠀⣰⠇⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⠁⠀⠀⠀⠀⢠⠟⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣀⣠⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⢀⡄⠀⠀⠀⡞⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢸⡏⠁⠀⠀⠀⠀⡈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠃⢀⡞⠀⠀⠀⡴⠃⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣼⠁⠀⠀⢠⡴⡼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠃⠀⡞⠁⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣞⡥⡄⠀⠀⢸⡿⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⡰⠁⢀⡞⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⡞⢸⣇⠀⠀⠀⠈⠀⠀⠀⢀⡰⠁⠀⠀⠀⢀⡼⠁⢀⡞⠀⠀⠀⣾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣼⠁⢀⡎⠀⠀⠀⠀⠀⠀⠀⡞⠁⠀⠀⠀⣠⠏⠀⣠⠋⠀⠀⢰⡶⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢠⠇⠀⡞⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠰⠃⡀⢹⡅⠀⠀⣰⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣼⠀⢰⠁⠀⠐⣄⠀⠀⡠⠎⠀⠀⠀⢀⡀⠀⠀⠛⠋⠀⢀⡴⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⢀⠇⠀⠀⠀⠈⠉⠉⠀⠀⠀⠀⢠⠎⠀⠀⠀⠀⠀⣰⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠸⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡴⠋⡠⠂⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠹⣦⡍⢀⠀⠀⠀⠀⠀⣠⠞⣠⠞⠁⢀⣴⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠈⠳⠾⠥⣀⣠⣔⣋⣵⣊⡤⠴⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

Look at me Morty, I'M ON GPU""")

    dataset_files=[]
    for dirpath, dirnames, filenames in os.walk(args.dataset_path):
        python_files = [file for file in filenames if file.endswith(".py")]
        if python_files:
            dataset_files.extend([os.path.join(os.getcwd(),dirpath, file) for file in python_files])

    for file_path in dataset_files:
        results = []
        print("\033[91m" + file_path + "\033[0m")
        result = get_model_output_inspector(file_path)
