import os
import pandas as pd

WEATHER_PATH = "data/raw_weather"
ELECTRICITY_PATH = "data/raw_electricity"
OUTPUT_FILE = "data/inspection_report.txt"


def inspect_folder(folder_path, dataset_name, report_lines):
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"INSPECTING {dataset_name.upper()} FOLDER")
    report_lines.append("=" * 80)

    if not os.path.exists(folder_path):
        report_lines.append(f"Folder not found: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not files:
        report_lines.append("No CSV files found.")
        return

    for file in files:
        file_path = os.path.join(folder_path, file)
        report_lines.append("\n" + "-" * 60)
        report_lines.append(f"FILE: {file}")
        report_lines.append("-" * 60)

        try:
            df = pd.read_csv(file_path)

            report_lines.append(f"Shape: {df.shape}")
            report_lines.append(f"Columns: {df.columns.tolist()}")
            report_lines.append("\nData Types:")
            report_lines.append(str(df.dtypes))

            report_lines.append("\nMissing Values:")
            report_lines.append(str(df.isna().sum()))

            # Date inspection
            date_cols = [col for col in df.columns if "date" in col.lower()]
            if date_cols:
                for col in date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        report_lines.append(f"\nDate Range ({col}):")
                        report_lines.append(f"Min: {df[col].min()}")
                        report_lines.append(f"Max: {df[col].max()}")
                    except:
                        report_lines.append(f"\nCould not parse {col} as datetime.")

            report_lines.append("\nFirst 5 Rows:")
            report_lines.append(str(df.head()))

        except Exception as e:
            report_lines.append(f"Error reading file: {e}")


if __name__ == "__main__":
    report_lines = []

    inspect_folder(WEATHER_PATH, "Raw Weather", report_lines)
    inspect_folder(ELECTRICITY_PATH, "Raw Electricity", report_lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nInspection report saved to: {OUTPUT_FILE}")