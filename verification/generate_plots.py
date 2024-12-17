import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate(dfs : [pd.DataFrame]):
    x = []

    imagined_recalled = []
    imagined_retold = []
    retold_recalled = []

    for idx, df in enumerate(dfs):
        # normalize the sequentiality values
        scaler = MinMaxScaler()
        df["scalar_text_sequentiality"] = pd.DataFrame(scaler.fit_transform(df[["scalar_text_sequentiality"]]),
                                                       columns=["sequentiality"])

        df['seq_type'] = df.apply(determine_bin, axis=1)

        # Calculate average sequentiality for each bin type
        df['seq_type'] = df.apply(determine_bin, axis=1)

        # Calculate average sequentiality for each bin type
        bin_averages = df.groupby('seq_type')['scalar_text_sequentiality'].mean()

        imagined_recalled_diff = percentage_dif(bin_averages['Imagined'], bin_averages['Recall'])
        imagined_retold_diff = percentage_dif(bin_averages['Imagined'], bin_averages['Retold'])
        retold_recalled_diff = percentage_dif(bin_averages['Retold'], bin_averages['Recall'])

        # Create tuple of the three bin averages
        pts = [(imagined_recalled_diff, imagined_retold_diff, retold_recalled_diff)]

        x.append(idx)
        for i, j, k in pts:
            imagined_recalled.append(i)
            imagined_retold.append(j)
            retold_recalled.append(k)

    plt.figure()
    plt.title("Prelim recreation")
    plt.xlabel("recall")
    plt.ylabel("% difference")

    plt.scatter(x, imagined_recalled, label="imagined vs. recalled", color="blue")
    plt.scatter(x, imagined_retold, label="imagined vs. retold", color="red")
    plt.scatter(x, retold_recalled, label="retold vs recalled", color="green")

    plt.legend(loc='upper right')
    plt.show()


def determine_bin(row):
    """Function used to divide each value into bins according to whether it was an imagined, or remembered"""
    # recAgnPairId = None for imagined stories -> ID of the retold story being recalled
    # if not None then its recalled

    # recImgPairId = None for retold stories -> ID of imagined story being recalled
    # if not None then its recalled

    if isinstance(row['recAgnPairId'], float) and isinstance(row['recImgPairId'], float):
        return 'Recall'  # This is a recall (neither points to it)
    elif not isinstance(row['recAgnPairId'], float) and isinstance(row['recImgPairId'], float):
        return 'Retold'  # This is a retold story with a recall
    elif isinstance(row['recAgnPairId'], float) and not isinstance(row['recImgPairId'], float):
        return 'Imagined'  # This is an imagined story with a recall
    else:
        return 'Error'  # Can't be both retold and imagined

def percentage_dif(l, r):
    numerator = abs(l - r)
    denominator = np.mean([l, r])

    if denominator == 0:
        denominator = 0.0000001

    return (numerator / denominator) * 100


if __name__ == "__main__":
    generate(pd.DataFrame())
