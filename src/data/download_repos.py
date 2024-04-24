import requests


def fetch_repos(since_date, access_token):
    # Prepare headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # API endpoint to search Python repositories created after the specified date
    url = f"https://api.github.com/search/repositories?q=created:>{since_date}+language:Python&sort=created&order=asc"

    # Make the API request
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["items"]
    else:
        print("Failed to fetch repositories")
        return None


def fetch_repo_contents(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def count_python_scripts(contents_url, headers):
    total_python_files = 0
    contents = fetch_repo_contents(contents_url, headers)

    if contents:
        for item in contents:
            if item["type"] == "file" and item["name"].endswith(".py"):
                total_python_files += 1
            elif item["type"] == "dir":
                # Recursively fetch and count Python scripts in subdirectories
                total_python_files += count_python_scripts(item["url"], headers)
    return total_python_files


def main(since_date, access_token):
    headers = {"Authorization": f"token {access_token}"}
    repos = fetch_repos(since_date, headers)

    if repos:
        for repo in repos:
            contents_url = repo["contents_url"].replace("{+path}", "")
            script_count = count_python_scripts(contents_url, headers)

            if 10 < script_count < 50:  # Check if the count is within the desired range
                print(f"Repository Name: {repo['name']}")
                print(f"Repository URL: {repo['html_url']}")
                print(f"Created Date: {repo['created_at']}")
                print(f"Python Script Count: {script_count}\n")


# Replace 'your_access_token' with your GitHub Personal Access Token
access_token = "token"

# Format YYYY-MM-DD
since_date = "2023-01-01"  # Adjust to the desired date

main(since_date, access_token)
