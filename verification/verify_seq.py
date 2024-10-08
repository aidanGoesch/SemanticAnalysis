import sys
import numpy as np
from src.sequentiality import SequentialityModel


def load_data():
    return [[]]


def write_data(file_name):
    pass


def verify_data(start : int, stop : int):
    sequentialities = []
    data = load_data()

    data_slice = data[start:stop]

    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a conversation with a doctor")

    for person in data_slice:
        seq = model.calculate_total_sequentiality(person.story)
        sequentialities.append(seq)

    write_data(f"{start}-{stop}.csv")


if __name__ == "__main__":
    verify_data(*sys.argv)
