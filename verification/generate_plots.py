import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_balanced_dataset(df, n_samples=130):
    """
    Create a balanced dataset with equal number of samples for each story type.

    Args:
        df: DataFrame from CSV file
        n_samples: Number of samples to keep for each type (default: 130)
    """
    # Combine all DataFrames
    combined_df = df

    # Add story type column
    combined_df['story_type'] = combined_df.apply(determine_bin, axis=1)

    # Initialize empty DataFrame for balanced data
    balanced_data = pd.DataFrame()

    # Sample equal number of stories from each type
    for story_type in ['Imagined', 'Retold', 'Recall']:
        type_data = combined_df[combined_df['story_type'] == story_type]

        # If we have more samples than needed, randomly sample
        if len(type_data) > n_samples:
            sampled_data = type_data.sample(n=n_samples, random_state=42)  # Set random_state for reproducibility
        else:
            print(f"Warning: Only {len(type_data)} samples available for {story_type} (needed {n_samples})")
            sampled_data = type_data

        balanced_data = pd.concat([balanced_data, sampled_data], ignore_index=True)

    # Save to CSV
    balanced_data.to_csv('truncated_data.csv', index=False)

    # Print final counts
    print("\nFinal dataset counts:")
    print(balanced_data['story_type'].value_counts())

    return balanced_data

def generate_2d(dfs : [pd.DataFrame]):
    x = []

    imagined_recalled = []
    imagined_retold = []
    retold_recalled = []

    for idx, df in enumerate(dfs):
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

    plt.plot(x, imagined_recalled, label="imagined vs. recalled", color="purple")
    plt.plot(x, imagined_retold, label="imagined vs. retold", color="green")
    plt.plot(x, retold_recalled, label="retold vs recalled", color="orange")

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


def standard_error(data):
    """Calculate the standard error of the mean"""
    return np.std(data, ddof=1) / np.sqrt(len(data))


def calculate_sequentiality_by_history(dfs):
    """
    Calculate average sequentiality values and print bin counts for each history length.
    """
    # Initialize lists for each type
    imagined = []
    retold = []
    recalled = []
    imagined_err = []
    retold_err = []
    recalled_err = []

    # Process each DataFrame (1-9)
    for i, df in enumerate(dfs):
        # Group stories by type
        df['story_type'] = df.apply(determine_bin, axis=1)

        # Print counts for this DataFrame
        counts = df['story_type'].value_counts()
        print(f"\nHistory Length {i + 1}:")
        for story_type in ['Imagined', 'Retold', 'Recall']:
            count = counts[story_type] if story_type in counts else 0
            print(f"{story_type}: {count} data points")

        # Calculate means and standard errors for each type
        for story_type, means, errors in zip(
                ['Imagined', 'Retold', 'Recall'],
                [imagined, retold, recalled],
                [imagined_err, retold_err, recalled_err]
        ):
            type_data = df[df['story_type'] == story_type]['scalar_text_sequentiality']
            if not type_data.empty:
                means.append(type_data.mean())
                errors.append(standard_error(type_data))
            else:
                means.append(None)
                errors.append(None)

    return [(imagined, imagined_err),
            (retold, retold_err),
            (recalled, recalled_err)]


def generate_2a(dfs):
    """Plot sequentiality values for different story types"""
    results = calculate_sequentiality_by_history(dfs)
    x = list(range(len(dfs)))  # 0-8 for the 9 DataFrames

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green']
    labels = ['imagined', 'retold', 'recalled']

    for (means, errors), color, label in zip(results, colors, labels):
        plt.errorbar(x, means, yerr=errors,
                     label=label, color=color,
                     marker='o', capsize=3)

    plt.xlabel('History Length')
    plt.ylabel('Sequentiality')
    plt.title('Sequentiality with Varying History Length')
    plt.legend()
    plt.show()


def generate_data_proportion_chart(file_path:str="./datasets/hcV3-stories.csv", title:str="Proportions of hcV3-stories.csv"):
    df = pd.read_csv(file_path)


    df['story_type'] = df.apply(determine_bin, axis=1)

    # Print counts for this DataFrame
    stats = df['story_type'].value_counts()
    print(stats)

    labels, counts = [], []
    for key, value in stats.items():
        labels.append(key)
        counts.append(value)
    
    plt.pie(counts, labels=labels)
    plt.title(title)
    plt.show()
    



# x = [5, 10, 15, 20, 25]
# # s = [96.15147966588847, 253.69168474990875, 218.6335731248837, 216.62307874998078, 258.85097066592425]
# # p = [42.29447991680354, 69.11600054102018, 100.3485505420249, 132.64719383278862, 153.53329075011425]

# p = [43.23337691696361, 73.38943483307958, 114.94705275003798, 144.4546077500563, 162.4089816659689]

# s = [43.75251183309592, 73.2175690419972, 105.70431220787577, 137.43758141691796, 161.05119587481022]


# seq1 = [1.1295, .8012, .9326, 1.0341, 1.0873]
# seq2 = [1.3333, 0.8272, 0.9314, 1.0584, 1.1323]

# # s1 = [132.3931, 107.2755, 161.2929]
# # s2 = []

# plt.figure()
# plt.plot(x, s, label="normal", color="blue")
# plt.plot(x, p, label="inference_mode", color="orange")
# # plt.plot(x, [percentage_dif(s1, s2) for s1, s2 in zip(s, p)])
# # plt.ylim(top=150, bottom=0)
# # plt.plot(x, s1, label="sequential w/ optimization", color="red")
# plt.legend()
# plt.show()
    
    


if __name__ == "__main__":
    generate_2d(pd.DataFrame())
