import json
import os

import tqdm

BLOCKS_PATH = os.path.join(os.getcwd(), "blocks")

results_jsonl = open(
    os.path.join(
        os.getcwd(),
        "run_results",
        "BlocksRun6",
        "results_block_6.jsonl",
    ),
    "r",
)
results_data = [line for line in results_jsonl]

error_count = 0
for data in tqdm.tqdm(results_data):
    try:
        data = json.loads(data)
        for entry_num, entry in enumerate(data):
            file_dir = entry["file_path"].split("/data", 1)[1].split("/")[1]
            # filename is the part that end in .py
            file_name = entry["file_path"].split("/")[-1]
            if not os.path.exists(
                os.path.join(BLOCKS_PATH, file_dir, file_name.replace(".py", ""))
            ):
                os.makedirs(
                    os.path.join(BLOCKS_PATH, file_dir, file_name.replace(".py", ""))
                )

            save_path = os.path.join(
                BLOCKS_PATH,
                file_dir,
                file_name.replace(".py", ""),
                f"func_num_{entry_num}_original_" + file_name,
            )
            save_path_block = os.path.join(
                BLOCKS_PATH,
                file_dir,
                file_name.replace(".py", ""),
                f"func_num_{entry_num}_generated_" + file_name,
            )

            file_path = entry["file_path"]
            prefix = entry["prefix"]
            suffix = entry["suffix"]
            model_output = entry["model_output"]

            with open(save_path, "w") as file:
                # remove the the empty scapces from the beginning of the prefix
                prefix = prefix.lstrip()
                file.write(prefix + suffix)

            with open(
                save_path_block,
                "w",
            ) as file:
                # remove prefix from model_output
                model_output = model_output.lstrip()
                file.write(model_output)
    except Exception as e:
        print(e)
        error_count += 1
        pass
print("error_count: ", error_count)
