import json, tqdm, os

BLOCKS_PATH = "/Users/ahura/Nexus/TWMC/src/blocks/"
results_jsonl = open("/Users/ahura/Nexus/TWMC/Runs/BlocksRun2/results_block.jsonl", "r")
results_data = [line for line in results_jsonl]

error_count = 0
for data in tqdm.tqdm(results_data):
    try:
        data = json.loads(data)
        for entry_num, entry in enumerate(data):
            file_dir = entry["file_path"].split("/data", 1)[1].split("/")[1]

            if not os.path.exists(BLOCKS_PATH + file_dir):
                os.makedirs(BLOCKS_PATH + file_dir)

            save_path = (
                BLOCKS_PATH
                + file_dir
                + "/"
                + f"func_num_{entry_num}"
                + entry["file_path"].split("/")[-1]
            )
            save_path_block = (
                BLOCKS_PATH
                + file_dir
                + "/"
                + f"func_num_{entry_num}"
                + entry["file_path"].split("/")[-1].split(".")[0]
                + "_block.py"
            )

            file_path = entry["file_path"]
            prefix = entry["prefix"]
            suffix = entry["suffix"]
            model_output = entry["model_output"]

            with open(save_path, "w") as file:
                file.write(prefix + suffix)

            with open(
                save_path_block,
                "w",
            ) as file:
                # remove prefix from model_output

                file.write(model_output.replace(prefix, ""))
    except Exception as e:
        print(e)
        error_count += 1
        pass
print("error_count: ", error_count)
