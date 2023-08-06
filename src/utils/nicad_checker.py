import os, time, tqdm, json

NICAD_DIR = os.path.join("/", "Users", "ahura", "Downloads", "NiCad-6.2")
WORKING_DIR = os.path.join(os.getcwd())
"/Users/ahura/Downloads/NiCad-6.2/"
# get the directory names to run Nicad for
directories = {
    k: {"blocks": [], "orginals": []}
    for k in os.listdir(os.path.join(os.getcwd(), "src", "blocks"))
}
# for each directory, get the list of files in it
for directory in directories.keys():
    directories[directory]["blocks"] = [
        i
        for i in os.listdir(os.path.join(os.getcwd(), "src", "blocks", directory))
        if "_block" in i
    ]
    directories[directory]["orginals"] = [
        i
        for i in os.listdir(os.path.join(os.getcwd(), "src", "blocks", directory))
        if "_block" not in i
    ]

# for each block file in directory
for directory in directories.keys():
    nicad_results = {k: "" for k in directories[directory]["blocks"]}
    # for each block file in directory
    for block_file in tqdm.tqdm(directories[directory]["blocks"]):
        # copt only the block file alongside all other files in the originals to ./Nicad6/systems/<directory_name>
        # print directory name in green
        print("\033[92m" + os.path.join(NICAD_DIR, "systems", directory) + "\033[0m")
        if not os.path.exists(os.path.join(NICAD_DIR, "systems", directory)):
            os.makedirs(os.path.join(NICAD_DIR, "systems", directory))

        os.system(
            f"cp {os.path.join(os.getcwd(),'src','blocks',directory,block_file)} {os.path.join(NICAD_DIR,'systems',directory,block_file)}"
        )
        for original_file in directories[directory]["orginals"]:
            os.system(
                f"cp {os.path.join(os.getcwd(),'src','blocks',directory,original_file)} {os.path.join(NICAD_DIR,'systems',directory,original_file)}"
            )
            # time.sleep(2)
        # run Nicad on the systems folder
        os.chdir(NICAD_DIR)
        os.system(
            f"arch -x86_64 sh {NICAD_DIR}/nicad6 blocks py {NICAD_DIR}/systems/{directory}"
        )
        os.chdir(WORKING_DIR)
        # open the html file
        html_files = [
            file
            for file in os.listdir(
                os.path.join(NICAD_DIR, "systems", f"{directory}_blocks-blind-clones")
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
            nicad_results[block_file] = f.read()

        # once done, remove the transferred files
    json.dump(
        nicad_results,
        open(os.path.join(WORKING_DIR, f"nicad_results_{directory}.json"), "w"),
    )
    # delete evetything in the systems folder

os.system(f"rm -rf {os.path.join(NICAD_DIR,'systems')}")
