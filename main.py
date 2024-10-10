from src.sequentiality import SequentialityModel
from RPE.reader import read_csv
import numpy as np
from verification.verify_seq import *


# use slice notation for scenes

# TODO: sliding window - only use the last 4 sentences

def calculate_cumulative_sequentiality(data: np.array, start: int = None,
                                       stop: int = None):  # TODO: Add this to the SequentialityModel class
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a description of a film")

    sequentialities = []

    if start is not None:
        data = data[start:stop]

    for i in range(len(data)):
        print("-------------------------------------------------------------------------------------------")
        print("text:" + " ".join(data[:i]))
        v = i >= 100
        seq = model.calculate_total_sequentiality(" ".join(data[:i + 1]),
                                                  v)  # calculate the sequentiality up until every scene
        sequentialities.append(seq)  # can sum or do something with them

        print(f"Sequentiality of scene {i + 1}: {seq}")
        print("-------------------------------------------------------------------------------------------")

    print(f"sequentialities = {sequentialities}")


def calculate_individual_sequentiality(data: np.array, start: int = None,
                                       stop: int = None):  # TODO: Add this to the SequentialityModel class
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a description of a film")

    sequentialities = []

    if start is not None:
        data = data[start:stop]

    for i in range(len(data)):
        print("-------------------------------------------------------------------------------------------")
        print("text:" + data[i])
        v = i >= 100
        seq = model.calculate_total_sequentiality(data[i], v)  # calculate the sequentiality up until every scene
        sequentialities.append(seq)  # can sum or do something with them

        print(f"Sequentiality of scene {i + 1}: {seq}")
        print("-------------------------------------------------------------------------------------------")

    print(f"sequentialities = {sequentialities}")


def main(*args, **kwargs):
    print(args)
    print(kwargs)


if __name__ == '__main__':
    # d = read_csv("RPE/preliminary_analysis/scene descriptions.csv")
    # calculate_cumulative_sequentiality(d)
    # calculate_individual_sequentiality(d)

    verify_data(int(sys.argv[1]), int(sys.argv[2]))
