from src.sequentiality import SequentialityModel
from RPE.reader import read_csv
import numpy as np

# use slice notation for scenes

# TODO: sliding window - only use the last 4 sentences

def calculate_cumulative_sequentiality(data : np.array, start : int = None, stop : int = None):  # TODO: Add this to the SequentialityModel class
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a description of a film")

    sequentialities = []

    if start is not None:
        data = data[start:stop]

    for i in range(len(data)):
        print("text:" + " ".join(data[:i]))
        seq = model.calculate_sequentiality(" ".join(data[:i + 1]))  # calculate the sequentiality up until every scene
        sequentialities.append(seq)  # can sum or do something with them

        print(f"Sequentiality of scene {i + 1}: {seq}")

if __name__ == '__main__':
    d = read_csv("RPE/preliminary_analysis/scene descriptions.csv")
    print(d[0])
    calculate_cumulative_sequentiality(d)
