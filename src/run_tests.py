import os
import unittest


def run_tests(test_directory):
    # Discover and run tests
    suite = unittest.TestLoader().discover(test_directory)
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    run_results_directory = os.path.join(os.getcwd(), "run_results")

    if not os.path.exists(run_results_directory):
        print("\033[93mCreating run_results directory\033[0m")
        os.makedirs(run_results_directory)

    assert_errors_file = os.path.join(run_results_directory, "assert_errors.txt")
    if not os.path.exists(assert_errors_file):
        print("\033[93mCreating run_results/assrt_errors.txt\033[0m")
        open(assert_errors_file, "w").close()

    generated_file = os.path.join(run_results_directory, "generated.txt")
    if not os.path.exists(generated_file):
        print("\033[93mCreating run_results/generated.txt\033[0m")
        open(generated_file, "w").close()

    processed_tokens_file = os.path.join(run_results_directory, "processed_tokens.txt")
    if not os.path.exists(processed_tokens_file):
        print("\033[93mCreating run_results/processed_tokens.txt\033[0m")
        open(processed_tokens_file, "w").close()

    too_many_tokens_file = os.path.join(run_results_directory, "too_many_tokens.txt")
    if not os.path.exists(too_many_tokens_file):
        print("\033[93mCreating run_results/too_many_tokens.txt\033[0m")
        open(too_many_tokens_file, "w").close()

    data_directory = os.path.join(os.getcwd(), "data")
    subdirectories = [
        subdir
        for subdir in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, subdir))
    ]
    python_files_found = False

    for subdir in subdirectories:
        subdir_path = os.path.join(data_directory, subdir)
        python_files = [
            file for file in os.listdir(subdir_path) if file.endswith(".py")
        ]
        if python_files:
            python_files_found = True
            break

    assert (
        python_files_found
    ), "No python files found in subdirectories of data directory. Check the dataset"

    run_tests("/store/travail/vamaj/TWMC/tests/")
