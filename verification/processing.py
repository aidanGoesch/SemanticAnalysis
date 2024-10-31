import pandas as pd
import pathlib

MASTER_PATH = "./data/calculated_values/main.csv"

def merge_csvs(data_dir : str):
    """Function that merges all CSVs in a gien directory"""

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


if __name__ == "__main__":
    merge_csvs("./data/calculated_values")