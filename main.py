from verification.verify_seq import *
from verification.generate_plots import generate
import pandas as pd




def calculate_cumulative_sequentiality(data: np.array, start: int = None, stop: int = None):
    """!!! NOT USED ON HPC3 !!!"""
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a description of a film")

    sequentialities = []

    if start is not None:
        data = data[start:stop]

    for i in range(len(data)):
        print("-------------------------------------------------------------------------------------------")
        print("text:" + " ".join(data[:i]))
        v = i >= 100
        seq = model.calculate_text_sequentiality(" ".join(data[:i + 1]),
                                                  v)  # calculate the sequentiality up until every scene
        sequentialities.append(seq)  # can sum or do something with them

        print(f"Sequentiality of scene {i + 1}: {seq}")
        print("-------------------------------------------------------------------------------------------")

    print(f"sequentialities = {sequentialities}")


def calculate_individual_sequentiality(data: np.array, start: int = None, stop: int = None):
    """!!! NOT USED ON HPC3 !!!"""
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a description of a film")

    sequentialities = []

    if start is not None:
        data = data[start:stop]

    for i in range(len(data)):
        print("-------------------------------------------------------------------------------------------")
        print("text:" + data[i])
        v = i >= 100
        seq = model.calculate_text_sequentiality(data[i], v)  # calculate the sequentiality up until every scene
        sequentialities.append(seq)  # can sum or do something with them

        print(f"Sequentiality of scene {i + 1}: {seq}")
        print("-------------------------------------------------------------------------------------------")

    print(f"sequentialities = {sequentialities}")

def main():
    """function that generates the graph from the calculated sequentiality values"""

    dfs = []
    for i in range(9):
        dfs.append(pd.read_csv(f"./data/calculated_values/{i + 1}/main.csv"))

    generate(dfs)


if __name__ == '__main__':
    # this is how it was run on hpc3 - function is in verification/verify_seq.py
    # verify_data(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    main()