import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate(df : pd.DataFrame):
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

    x = [1]

    imagined_recalled = []
    imagined_retold = []
    retold_recalled = []

    for i, j, k in pts:
        imagined_recalled.append(i)
        imagined_retold.append(j)
        retold_recalled.append(k)

    plt.figure()
    plt.title("Prelim recreation")
    plt.xlabel("normalized sequentiality")
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

def get_pts(df : pd.DataFrame, verbose : bool = False):
    """This is wrong and where the problem is"""
    thresholds = [0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888, 1]
    labels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    bins = df['main_bins'] = pd.cut(df['scalar_text_sequentiality'], bins=thresholds, labels=labels,
                                    include_lowest=True)

    if verbose:
        bin_counts = bins.value_counts()
        print("Total number of values in each bin:")
        print(bin_counts)

    # recAgnPairId = None for imagined stories -> ID of the retold story being recalled
    # if not None then its recalled

    # recImgPairId = None for retold stories -> ID of imagined story being recalled
    # if not None then its recalled

    df['sub_bins'] = df.apply(determine_bin, axis=1)

    if verbose:
        sub_bin_counts = df['sub_bins'].value_counts()
        print("\nCount of values in each sub-bin:")
        print(sub_bin_counts)

    grouped_means = df.groupby(['main_bins', 'sub_bins'])["scalar_text_sequentiality"].mean()
    grouped_means = grouped_means.fillna(0)

    percentage_differences = {}

    for main_bin in labels:
        if main_bin in grouped_means.index.get_level_values(0):
            # Get the means for the current main bin
            means = grouped_means.loc[main_bin]

            # Extract the values for each sub-bin
            both_none_mean = grouped_means.get((main_bin, 'Both None'), 0)
            a_present_b_none_mean = grouped_means.get((main_bin, 'recAgnPairId Not None'), 0)
            b_present_a_none_mean = grouped_means.get((main_bin, 'recImgPairId Not None'), 0)

            if verbose:
                print(f"values: {both_none_mean}, {a_present_b_none_mean}, {b_present_a_none_mean}")
                print(means)

            # Calculate the percentage differences
            diff1 = percentage_dif(a_present_b_none_mean, both_none_mean)
            diff2 = percentage_dif(b_present_a_none_mean, a_present_b_none_mean)
            diff3 = percentage_dif(both_none_mean, b_present_a_none_mean)

            # Store the tuple in the dictionary
            percentage_differences[main_bin] = (diff1, diff2, diff3)

    if verbose:
        # Print the percentage differences for each main bin
        print("\nPercentage differences for each main bin:")
        for bin_label, diffs in percentage_differences.items():
            print(f"{bin_label}: {diffs}")

    return percentage_differences




if __name__ == "__main__":
    generate(pd.DataFrame())
