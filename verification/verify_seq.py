import sys
import numpy as np
import pandas as pd
from src.sequentiality import SequentialityModel


def load_data():
    df = pd.read_csv("./data/hcV3-stories.csv")
    return df


def write_data(file_name, data : pd.DataFrame):
    data.to_csv(f"./data/calculated_values/{file_name}")


def verify_data(partition_id, participant_id):
    sequentialities = pd.DataFrame(columns=["sequentiality"])
    data = load_data()

    data_slice = data.iloc[partition_id + participant_id]

    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a conversation with a doctor")

    seq = model.calculate_total_sequentiality(data_slice.story)
    sequentialities.append(seq)
    sequentialities.loc[0] = "this is a test"

    write_data(f"{partition_id + participant_id}.csv", sequentialities)


if __name__ == "__main__":
    # verify_data(*sys.argv)
    load_data()
