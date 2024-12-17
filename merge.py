import pandas as pd
import pathlib
import sys
import os

MASTER_PATH = "./data/calculated_values/main.csv"

def merge_csvs(data_dir : str):
    """
    Function that merges all CSVs in a gien directory

    :param directory: Path to the directory containing CSV files
    """

    main_df = pd.DataFrame(columns=["AssignmentId",
                                            "scalar_text_sequentiality",
                                            "sentence_total_sequentialities",
                                            "sentence_contextual_sequentialities",
                                            "sentence_topic_sequentialities",
                                            "story",
                                            "recAgnPairId",
                                            "recImgPairId"])

    for file in pathlib.Path(data_dir).iterdir():
        tmp_df = pd.read_csv(file)

        main_df = pd.concat([main_df, tmp_df])

    main_df.to_csv(MASTER_PATH)


def delete_csvs_except_main(directory):
    """
    Delete all CSV files in the specified directory except 'main.csv'

    :param directory: Path to the directory containing CSV files
    """
    # Ensure directory path ends with a slash
    directory = directory.rstrip('/') + '/'

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV and not 'main.csv'
        if filename.endswith('.csv') and filename != 'main.csv':
            filepath = os.path.join(directory, filename)
            try:
                os.remove(filepath)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")


if __name__ == "__main__":
    """
    example usage:
    python3 merge.py ./data/directory
    """
    path = str(sys.argv[1])
    merge_csvs(path)
    print("Merge complete")
    delete_csvs_except_main(path)
