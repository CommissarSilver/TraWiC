import os, json
from bs4 import BeautifulSoup


def parse_clone_classes_and_files(html_content):
    """
    Parses the given HTML content to extract clone classes and file names.

    :param html_content: A string containing the HTML content.
    :return: A dictionary containing the extracted clone classes and file names.
             The keys are in the format 'clone_class_X', where X is the index of the clone class,
             and the values are lists of file names within that clone class.
    """
    # Initialize the dictionary to store the results
    clone_classes_dict = {}

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all tables that contain clone classes and file names
    tables = soup.find_all("table", border="2", cellpadding="4")

    # Iterate through the tables and extract the information
    for clone_class_index, table in enumerate(tables, start=1):
        # Extract all file names within this clone class
        file_links = table.find_all("a")
        file_names = [link.text.strip() for link in file_links]

        # Add to the dictionary using the clone class index as the key
        clone_classes_dict[f"clone_class_{clone_class_index}"] = file_names

    return clone_classes_dict


# Directory containing the original JSON files
original_directory_path = "/Users/ahura/Nexus/TWMC/nicad_results/original"
# Directory to save the result JSON files
results_directory_path = "/Users/ahura/Nexus/TWMC/nicad_results/results"

# Create the results directory if it doesn't exist
os.makedirs(results_directory_path, exist_ok=True)

# Iterate over all the files in the original directory
for filename in os.listdir(original_directory_path):
    if filename.endswith(".json"):
        with open(os.path.join(original_directory_path, filename), "r") as f:
            nicad_results = json.load(f)
            html_content = list(nicad_results.values())[0]

        # Parse the HTML content and get the result
        result_dict = parse_clone_classes_and_files(html_content)

        # Save the result as a JSON file in the results directory
        json_file_path = os.path.join(
            results_directory_path, filename.replace(".json", "_result.json")
        )
        with open(json_file_path, "w") as json_file:
            json.dump(result_dict, json_file)

        print(f"Processed {filename} and saved results to {json_file_path}")
