import json
import os


def get_repos_list():
    with open("/Users/ahvra/Nexus/TWMC/repos.json", "r") as f:
        repos = json.load(f)
    with open("/Users/ahvra/Nexus/TWMC/discard.json", "r") as f:
        discard_repos = json.load(f)
    merged_repos = repos + discard_repos
    return merged_repos


def clone_repos(repos, save_dir):
    for repo in repos:
        print(repo["repostory_name"])
        # use git git clone repo['repository_url'] to clone the repo in the terminal to the given directory
        os.system(
            f"git clone {repo['repostory_url']} {save_dir}/{repo['repostory_name']}"
        )


l = get_repos_list()
clone_repos(l, "/Users/ahvra/Nexus/TWMC/repos")
print("hi")
