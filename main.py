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
        print("-------------------------------------------------------------------------------------------")
        print("text:" + " ".join(data[:i]))
        v = i >= 100
        seq = model.calculate_total_sequentiality(" ".join(data[:i + 1]), v)  # calculate the sequentiality up until every scene
        sequentialities.append(seq)  # can sum or do something with them

        print(f"Sequentiality of scene {i + 1}: {seq}")
        print("-------------------------------------------------------------------------------------------")

    print(f"sequentialities = {sequentialities}")

if __name__ == '__main__':
    # d = read_csv("RPE/preliminary_analysis/scene descriptions.csv")
    # print(d[0])
    # calculate_cumulative_sequentiality(d)

    # calculate values
    sequentialities = [0.8335433668883828, 1.9963361268086237, 1.9188129317826614, 1.779050758289371,
                       1.9307388874203304, 1.893751055346082, 0.36445321924398394, -1.5996239443578826,
                       -1.312597324725007, -1.006148255485575, -0.8669968819705246, -0.6650615208508861,
                       -1.6316238371971092, -1.080335940027522, -1.0292703662440599, -0.7967674830560237,
                       -0.695877374357641, -0.6836993418557638, -0.6550893884038461, -0.5380177053654409,
                       -0.48090277512863994, -1.0423730740206114, -1.030656261164259, -1.7969003331125943,
                       -1.722155753512521, -1.6579208910789862]


    from matplotlib import pyplot as plt

    plt.scatter(np.arange(26), sequentialities)
    plt.plot(np.arange(26), sequentialities)
    plt.show()

