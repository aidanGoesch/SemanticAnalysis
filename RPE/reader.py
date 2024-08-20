import numpy as np
import csv

def read_csv(file_path : str) -> np.array:
    raw = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        raw = np.array(list(reader))

    ret = ["" for _ in range(26)]  # create an empty list the same length as the number of scenes

    for row in raw:
        ret[int(row[0]) - 1] += " " + row[2]

    return ret


if __name__ == '__main__':
    tmp = read_csv("./preliminary_analysis/scene descriptions.csv")

    for row in tmp:
        print(row)
