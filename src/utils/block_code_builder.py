import json, tqdm

results_jsonl = open("/Users/ahura/Nexus/TWMC/results_block.jsonl", "r")
results_data = [line for line in results_jsonl]


for data in tqdm.tqdm(results_data):
    data = json.loads(data)
    for entry in data:
        file_path = entry["file_path"]
        prefix = entry["prefix"]
        suffix = entry["suffix"]
        model_output = entry["model_output"]
        with open(
            "/Users/ahura/Nexus/TWMC/src/blocks/" + file_path.split("/")[-1], "w"
        ) as file:
            file.write(prefix + suffix)
        with open(
            "/Users/ahura/Nexus/TWMC/src/blocks/" + "block_"+
            file_path.split("/")[-1],
            "w",
        ) as file:
            file.write(model_output)
print("hi")
