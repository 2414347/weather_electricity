# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
# This section imports required Python libraries used for:
#
# os:
#   Used to interact with the operating system.
#   Helps check folder existence and list files.
#
# pandas:
#   Used to read CSV files and inspect dataset content.

import os
import pandas as pd


# ============================================================
# DEFINE INPUT AND OUTPUT PATHS
# ============================================================
# This section defines folder locations where datasets exist
# and where the inspection report will be saved.
#
# WEATHER_PATH:
#   Folder containing raw weather CSV files.
#
# ELECTRICITY_PATH:
#   Folder containing raw electricity demand CSV files.
#
# OUTPUT_FILE:
#   Text file where the inspection results will be saved.

WEATHER_PATH = "data/raw_weather"
ELECTRICITY_PATH = "data/raw_electricity"
OUTPUT_FILE = "data/inspection_report.txt"


# ============================================================
# FUNCTION TO INSPECT DATA FOLDERS
# ============================================================
# This function inspects all CSV files inside a folder
# and collects useful information about each dataset.
#
# Parameters:
#
# folder_path:
#   Path of the folder to inspect.
#
# dataset_name:
#   Name used for labeling the dataset section
#   inside the report.
#
# report_lines:
#   A list used to store all inspection results
#   before writing them to a file.
#
# Tasks performed by this function:
#
# 1. Check if folder exists.
# 2. Find all CSV files.
# 3. Load each CSV file.
# 4. Extract dataset details:
#       - Shape (rows, columns)
#       - Column names
#       - Data types
#       - Missing values
#       - Date ranges
#       - First 5 rows preview
#
# All results are saved into report_lines list.

def inspect_folder(folder_path, dataset_name, report_lines):

    # Add section header to report
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"INSPECTING {dataset_name.upper()} FOLDER")
    report_lines.append("=" * 80)

    # --------------------------------------------------------
    # CHECK IF FOLDER EXISTS
    # --------------------------------------------------------
    # If the folder does not exist,
    # record the message and stop processing.

    if not os.path.exists(folder_path):
        report_lines.append(f"Folder not found: {folder_path}")
        return

    # --------------------------------------------------------
    # FIND ALL CSV FILES
    # --------------------------------------------------------
    # List all files ending with ".csv"

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # If no CSV files found, record message
    if not files:
        report_lines.append("No CSV files found.")
        return

    # --------------------------------------------------------
    # PROCESS EACH CSV FILE
    # --------------------------------------------------------
    # Loop through every CSV file
    # and inspect its content.

    for file in files:

        # Construct full file path
        file_path = os.path.join(folder_path, file)

        # Add file header to report
        report_lines.append("\n" + "-" * 60)
        report_lines.append(f"FILE: {file}")
        report_lines.append("-" * 60)

        try:

            # ------------------------------------------------
            # LOAD CSV FILE
            # ------------------------------------------------
            # Read CSV into pandas DataFrame.

            df = pd.read_csv(file_path)

            # ------------------------------------------------
            # BASIC DATASET INFORMATION
            # ------------------------------------------------
            # Shape shows:
            #   Number of rows
            #   Number of columns

            report_lines.append(f"Shape: {df.shape}")

            # Column names list
            report_lines.append(f"Columns: {df.columns.tolist()}")

            # Display data types
            report_lines.append("\nData Types:")
            report_lines.append(str(df.dtypes))

            # ------------------------------------------------
            # MISSING VALUE ANALYSIS
            # ------------------------------------------------
            # Count number of missing values
            # in each column.

            report_lines.append("\nMissing Values:")
            report_lines.append(str(df.isna().sum()))

            # ------------------------------------------------
            # DATE COLUMN INSPECTION
            # ------------------------------------------------
            # Automatically detect columns
            # containing the word "date".

            date_cols = [
                col for col in df.columns
                if "date" in col.lower()
            ]

            # If date columns found
            if date_cols:

                for col in date_cols:

                    try:

                        # Convert column to datetime
                        df[col] = pd.to_datetime(df[col])

                        # Display date range
                        report_lines.append(f"\nDate Range ({col}):")

                        report_lines.append(
                            f"Min: {df[col].min()}"
                        )

                        report_lines.append(
                            f"Max: {df[col].max()}"
                        )

                    except:

                        # If conversion fails
                        report_lines.append(
                            f"\nCould not parse {col} as datetime."
                        )

            # ------------------------------------------------
            # DISPLAY SAMPLE DATA
            # ------------------------------------------------
            # Show first 5 rows of dataset
            # to preview structure.

            report_lines.append("\nFirst 5 Rows:")
            report_lines.append(str(df.head()))

        except Exception as e:

            # ------------------------------------------------
            # ERROR HANDLING
            # ------------------------------------------------
            # If file reading fails,
            # record error message.

            report_lines.append(f"Error reading file: {e}")


# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================
# This section runs when the script is executed.
#
# Steps performed:
#
# 1. Create empty report list.
# 2. Inspect weather folder.
# 3. Inspect electricity folder.
# 4. Save inspection report to text file.
# 5. Display confirmation message.

if __name__ == "__main__":

    # Create empty list to store report
    report_lines = []

    # Inspect weather dataset folder
    inspect_folder(
        WEATHER_PATH,
        "Raw Weather",
        report_lines
    )

    # Inspect electricity dataset folder
    inspect_folder(
        ELECTRICITY_PATH,
        "Raw Electricity",
        report_lines
    )

    # --------------------------------------------------------
    # SAVE REPORT TO FILE
    # --------------------------------------------------------
    # Write collected inspection results
    # into a text file.

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(report_lines))

    # Display completion message
    print(
        f"\nInspection report saved to: {OUTPUT_FILE}"
    )
