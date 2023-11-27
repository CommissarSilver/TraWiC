import pandas as pd


def format_latex_table(csv_data):
    """
    Function to format the CSV data into a LaTeX table.

    :param csv_data: DataFrame containing the CSV data.
    :return: String containing the formatted LaTeX table.
    """
    # LaTeX table start
    latex_table = "\\begin{table}\n"
    latex_table += "    \\resizebox{\\textwidth}{!}{\n"
    latex_table += "        \\centering\n"
    latex_table += "        \\begin{tabular}{c c r r r r r}\n"
    latex_table += "            \\toprule\n"
    latex_table += "            Edit Distance Threshold & Repository Inclusion Criterion & Precision(\\%) & Accuracy(\\%) & F-score(\\%) & Sensitivity(\\%) & Specificity (\\%)\\\\\\\n"
    latex_table += "            \\midrule\n"

    # Iterate over the rows of the CSV data and format them for LaTeX
    for index, row in csv_data.iterrows():
        # Apply multirow formatting for semantic and syntactic thresholds if necessary
        if index == 0 or (
            csv_data.loc[index, "semantic threshold"]
            != csv_data.loc[index - 1, "semantic threshold"]
        ):
            num_rows = len(
                csv_data[csv_data["semantic threshold"] == row["semantic threshold"]]
            )
            latex_table += f"                \\multirow{{{num_rows}}}{{*}}{{\\textbf{{{row['semantic threshold']}}}}} & "
        else:
            latex_table += "                & "

        # Adding the rest of the data
        latex_table += f"{row['inclusion criterion']} & {row['precision']*100:.2f} & {row['accuracy']*100:.2f} & {row['f1-score']*100:.2f} & {row['sensitivity']*100:.2f} & {row['specificity']*100:.2f}\\\\\n"

    # LaTeX table end
    latex_table += "            \\bottomrule\n"
    latex_table += "        \\end{tabular}\n"
    latex_table += "    }\n"
    latex_table += "\\end{table}"

    return latex_table


csv_data = pd.read_csv("/store/travail/vamaj/TWMC/clf_xgb.csv")
# Generate the LaTeX table
latex_table_content = format_latex_table(csv_data)
with open("sample.tex", "w") as file:
    file.write(latex_table_content)


# Display the first 1000 characters of the generated LaTeX table for brevity
