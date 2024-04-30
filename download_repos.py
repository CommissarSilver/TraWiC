import json
import os


def get_repos_list():
    with open(os.path.join(os.getcwd(), "repos.json"), "r") as f:
        repos = json.load(f)
    with open(os.path.join(os.getcwd(), "discard.json"), "r") as f:
        discard_repos = json.load(f)
    merged_repos = repos + discard_repos
    return merged_repos


def clone_repos(repos, save_dir):
    for repo in repos:
        print(repo["repostory_name"])

        os.system(
            f"git clone {repo['repostory_url']} {save_dir}/{repo['repostory_name']}"
        )


l = get_repos_list()
clone_repos(
    l,
    os.path.join(
        os.getcwd(),
        "data",
        "repos",
    ),
)
print("hi")
