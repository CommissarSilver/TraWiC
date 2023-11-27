import pandas as pd


# def format_latex_table(csv_data):
#     """
#     Function to format the CSV data into a LaTeX table.

#     :param csv_data: DataFrame containing the CSV data.
#     :return: String containing the formatted LaTeX table.
#     """
#     # LaTeX table start
#     latex_table = "\\begin{table}\n"
#     latex_table += "    \\resizebox{\\textwidth}{!}{\n"
#     latex_table += "        \\centering\n"
#     latex_table += "        \\begin{tabular}{c c c r r r r r}\n"
#     latex_table += "            \\toprule\n"
#     latex_table += "            Edit Distance Threshold & Noise Ratio & Repository Inclusion Criterion & Precision(\\%) & Accuracy(\\%) & F-score(\\%) & Sensitivity(\\%) & Specificity (\\%)\\\\\\\n"
#     latex_table += "            \\midrule\n"

#     # Iterate over the rows of the CSV data and format them for LaTeX
#     for index, row in csv_data.iterrows():
#         # Apply multirow formatting for semantic and syntactic thresholds if necessary
#         if index == 0 or (
#             csv_data.loc[index, "semantic_threshold"]
#             != csv_data.loc[index - 1, "semantic_threshold"]
#         ):
#             num_rows = len(
#                 csv_data[csv_data["semantic_threshold"] == row["semantic_threshold"]]
#             )
#             latex_table += f"                \\multirow{{{num_rows}}}{{*}}{{\\textbf{{{row['semantic_threshold']}}}}} & "
#         else:
#             latex_table += "                & "

#         # Adding the rest of the data
#         latex_table += f"{row['sensitivity_threshold']} & {row['level']} & {row['precision']*100:.2f} & {row['accuracy']*100:.2f} & {row['f1']*100:.2f} & {row['recall']*100:.2f} & {row['specificity']*100:.2f}\\\\\n"

#     # LaTeX table end
#     latex_table += "            \\bottomrule\n"
#     latex_table += "        \\end{tabular}\n"
#     latex_table += "    }\n"
#     latex_table += "\\end{table}"

#     return latex_table


# csv_data = pd.read_csv("/store/travail/vamaj/TWMC/results_semantic_sensitivity.csv")
# # Generate the LaTeX table
# latex_table_content = format_latex_table(csv_data)
# with open("sample2.tex", "w") as file:
#     file.write(latex_table_content)


# Display the first 1000 characters of the generated LaTeX table for brevity
def format_latex_table_svm(csv_data):
    """
    Function to format the CSV data into a LaTeX table for the SVM dataset.

    :param csv_data: DataFrame containing the CSV data.
    :return: String containing the formatted LaTeX table.
    """
    # LaTeX table start
    latex_table = "\\begin{table}\n"
    latex_table += "    \\centering\n"
    latex_table += "    \\begin{tabular}{c c c r r r r r}\n"
    latex_table += "        \\toprule\n"
    latex_table += "        Edit Distance Threshold & Noise Ratio & Repository Inclusion Criterion & Precision(\\%) & Accuracy(\\%) & F-score(\\%) & Sensitivity(\\%) & Specificity (\\%)\\\\\n"
    latex_table += "        \\midrule\n"

    # Track the last seen values for 'semantic threshold' and 'syntactic threshold' for multirow
    last_semantic = None
    last_syntactic = None
    for index, row in csv_data.iterrows():
        # Semantic threshold multirow check
        if row["semantic_threshold"] != last_semantic:
            num_semantic_rows = len(
                csv_data[csv_data["semantic_threshold"] == row["semantic_threshold"]]
            )
            latex_table += f"            \\multirow{{{num_semantic_rows}}}{{*}}{{\\textbf{{{row['semantic_threshold']}}}}} "
            last_semantic = row["semantic_threshold"]
            first_syntactic_in_group = True
        else:
            latex_table += "            "
            first_syntactic_in_group = False

        # Syntactic threshold multirow check
        if first_syntactic_in_group or row["sensitivity_threshold"] != last_syntactic:
            num_syntactic_rows = len(
                csv_data[
                    (csv_data["semantic_threshold"] == row["semantic_threshold"])
                    & (
                        csv_data["sensitivity_threshold"]
                        == row["sensitivity_threshold"]
                    )
                ]
            )
            latex_table += f"& \\multirow{{{num_syntactic_rows}}}{{*}}{{{row['sensitivity_threshold']}}} "
            last_syntactic = row["sensitivity_threshold"]
        else:
            latex_table += "& "

        # Adding the rest of the data
        latex_table += f"& {row['level']} & {row['precision']*100:.2f} & {row['accuracy']*100:.2f} & {row['f1']*100:.2f} & {row['recall']*100:.2f} & {row['specificity']*100:.2f}\\\\\n"

    # LaTeX table end
    latex_table += "        \\bottomrule\n"
    latex_table += "    \\end{tabular}\n"
    latex_table += "\\end{table}"

    return latex_table


csv_data = pd.read_csv("/store/travail/vamaj/TWMC/results_combined.csv")
# Apply the function to the loaded CSV data
latex_table_svm_content = format_latex_table_svm(csv_data)
with open("sampleSen_comb.tex", "w") as file:
    file.write(latex_table_svm_content)
