import sys
import numpy as np
import pandas as pd
from src.sequentiality import SequentialityModel


def load_data():
    df = pd.read_csv("./data/hcV3-stories.csv")
    return df


def write_data(file_name, data : pd.DataFrame):
    data.to_csv(f"./data/calculated_values/{file_name}")


def verify_data(start : int, stop : int):
    sequentialities = pd.DataFrame(columns=["sequentiality"])
    data = load_data()

    data_slice = data[start:stop]

    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a conversation with a doctor")

    for i in range(len(data_slice)):
        seq = model.calculate_total_sequentiality(data.iloc[i].story)
        sequentialities.append(seq)
        sequentialities.loc[i] = "this is a test"

    write_data(f"{start}-{stop}.csv", sequentialities)


if __name__ == "__main__":
    # verify_data(*sys.argv)
    load_data()
