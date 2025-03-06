from verification.verify_seq import *
from verification.generate_plots import generate_2d, generate_2a, create_balanced_dataset, generate_data_proportion_chart, percentage_dif
from verification.subset import analyze_embeddings, save_top_stories, merge_top_stories, determine_bin, make_large_subset, make_proportional_subset_using_other_subset
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
import gc
import os


# Models:

"SakanaAI/TinySwallow-1.5B-Instruct"
"meta-llama/Llama-3.3-70B-Instruct"
"microsoft/Phi-3-mini-4k-instruct"
"neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8"


# HPC Checklist
#   - Is the model name correct?
#   - Is the save file location correct?
#   - Is the dataset file path correct?
#   - Is the correct function being run with the correct arguments in main.py?
#   - Is the version of the code on HPC what you want to run?


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

def generate_plots(data_path:str="./outputs/phi-4k-mini", file_name:str="main.csv"):
    """
    Function that generates the graph from the calculated sequentiality values. Takes as an argument the path to where the data is stored, and the filename
    """

    dfs = []
    for i in range(9):
        dfs.append(pd.read_csv(f"{data_path}/{i + 1}/{file_name}"))

    generate_2a(dfs)
    generate_2d(dfs)

def explore_random_seeds():
    s = []
    for i in range(30):
        torch.manual_seed(i)  # Add before model initialization
        np.random.seed(i)
        print(f"============{i} started============")
        model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a short story")
        s.append(model.calculate_text_sequentiality(
            "Concerts are my most favorite thing, and my boyfriend knew it. That's why, for our anniversary, he got me tickets to see my favorite artist. Not only that, but the tickets were for an outdoor show, which I love much more than being in a crowded stadium. Since he knew I was such a big fan of music, he got tickets for himself, and even a couple of my friends. He is so incredibly nice and considerate to me and what I like to do. I will always remember this event and I will always cherish him. On the day of the concert, I got ready, and he picked me up and we went out to a restaurant beforehand. He is so incredibly romantic. He knew exactly where to take me without asking. We ate, laughed, and had a wonderful dinner date before the big event. We arrived at the concert and the music was so incredibly beautiful. I loved every minute of it. My friends, boyfriend, and I all sat down next to each other. As the music was slowly dying down, I found us all getting lost just staring at the stars. It was such an incredibly unforgettable and beautiful night.")[
                     0])

        del model
        gc.collect()
        print(f"sequentiality with seed {i}: {s[-1]}")

    plt.figure()
    plt.scatter([1 for _ in range(30)], s)
    plt.show()

    print(s)

def print_model_comparisons():
    """Function that will print model comparison data stored in a csv"""
    df = pd.read_csv("./outputs/misc/comparison_data_35804930.csv")

    # print(np.mean(df["small_model_times"].iloc[0]))

    small_model_times = eval(df["small_model_times"].iloc[0])
    big_model_times = eval(df["big_model_times"].iloc[0])

    print(f"average execution time of small model: {np.mean(small_model_times)} with standard deviation {np.std(small_model_times)}")
    print(f"70b quantized model takes on average {np.mean(big_model_times) / np.mean(small_model_times)}x longer than the 4b model")

    small_model_seq = [x[0] for x in eval(df["small_model_seq"].iloc[0])]
    big_model_seq = [x[0] for x in eval(df["big_model_seq"].iloc[0])]

    print(big_model_seq)

    print(f"small model sequentiality values: {small_model_seq}\n\taverage: {np.mean(small_model_seq)}\n\tstandard deviation: {np.std(small_model_seq)}")
    print(f"big model sequentiality values: {big_model_seq}\n\taverage: {np.mean(big_model_seq)}\n\tstandard deviation: {np.std(big_model_seq)}")

def create_mini_files(base_path="./outputs/phi-4k-mini", merged_file="./datasets/hcV3-stories-1000.csv"):
    """
    Creates main-mini.csv files in each numbered folder, containing only the rows
    that match AssignmentIDs from the merged top stories file.
    
    Args:
        base_path (str): Path to the calculated_values directory
        merged_file (str): Path to the merged top stories file
    """
    # Read the merged file to get our target AssignmentIDs
    try:
        merged_df = pd.read_csv(merged_file)
        target_ids = set(merged_df['AssignmentId'])
        print(f"Found {len(target_ids)} unique AssignmentIDs in merged file")
    except FileNotFoundError:
        print(f"Error: Could not find {merged_file}")
        return
    
    # Process each numbered folder
    for folder in range(1, 10):  # Folders 1 through 9
        folder_path = os.path.join(base_path, str(folder))
        main_csv_path = os.path.join(folder_path, "main.csv")
        output_path = os.path.join(folder_path, "main-mini.csv")
        
        try:
            # Read the main.csv file
            df = pd.read_csv(main_csv_path)
            
            # Filter for only the rows with matching AssignmentIDs
            mini_df = df[df['AssignmentId'].isin(target_ids)]
            
            # Save the filtered data
            mini_df.to_csv(output_path, index=False)
            
            print(f"Created main-mini.csv in folder {folder} with {len(mini_df)} rows")
            
        except FileNotFoundError:
            print(f"Warning: Could not find main.csv in folder {folder}")
            continue
        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
            continue

def find_representative_samples(base_path: str = "./outputs/", samples_per_folder: int = 15):
    """
    Finds representative samples from each CSV by selecting stories closest to the mean
    of numerical values.
    """
    all_representative_samples = []
    
    # Process each folder
    for folder_num in range(1, 10):
        folder_path = os.path.join(base_path, str(folder_num))
        csv_path = os.path.join(folder_path, "main.csv")
        
        try:
            # Read the CSV
            df = pd.read_csv(csv_path)
            print(f"\nProcessing folder {folder_num}")
            print(f"Found {len(df)} rows")
            
            # Get numeric columns, excluding any metadata or index columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols 
                          if not col.startswith('Unnamed') 
                          and col != 'AssignmentId']
            
            print(f"Using numeric columns: {numeric_cols}")
            
            if not numeric_cols:
                print(f"No numeric columns found in folder {folder_num}")
                continue
            
            # Calculate means of numeric columns
            means = df[numeric_cols].mean()
            
            # Calculate distances from mean for each row
            distances = []
            for idx, row in df.iterrows():
                try:
                    # Calculate Euclidean distance from mean
                    distance = np.sqrt(sum((row[numeric_cols] - means) ** 2))
                    distances.append({'index': idx, 'distance': distance})
                except Exception as e:
                    print(f"Error calculating distance for row {idx}: {str(e)}")
                    continue
            
            # Sort by distance and get closest samples
            sorted_distances = sorted(distances, key=lambda x: x['distance'])
            closest_indices = [d['index'] for d in sorted_distances[:samples_per_folder]]
            
            # Get the representative samples
            representative_samples = df.loc[closest_indices].copy()
            representative_samples['distance_from_mean'] = [d['distance'] 
                                                         for d in sorted_distances[:samples_per_folder]]
            representative_samples['source_folder'] = folder_num
            
            # Add to our collection
            all_representative_samples.append(representative_samples)
            
            print(f"Selected {len(representative_samples)} representative samples")
            
        except FileNotFoundError:
            print(f"Warning: Could not find main.csv in folder {folder_num}")
            continue
        except Exception as e:
            print(f"Error processing folder {folder_num}: {str(e)}")
            continue
    
    # Combine all representative samples
    if all_representative_samples:
        final_df = pd.concat(all_representative_samples, ignore_index=True)
        
        # Save the combined results
        output_file = "representative_samples.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(final_df)} representative samples to {output_file}")
        
        # Print summary statistics
        print("\nSummary of representative samples:")
        print(f"Total samples: {len(final_df)}")
        print("\nSamples per folder:")
        print(final_df['source_folder'].value_counts().sort_index())
        print("\nAverage distance from mean by folder:")
        print(final_df.groupby('source_folder')['distance_from_mean'].mean().sort_index())
        
        return final_df
    else:
        print("No representative samples were found")
        return None
    
def select_stories_near_mean(dataset_path, main_path, output_path):
    """
    Selects 120 stories for each story type using weighted probability sampling.
    Stories closer to the mean have higher probability of being selected.
    
    Args:
        dataset_path (str): Path to the original dataset (hcV3-stories.csv)
        main_path (str): Path to the main.csv file with model outputs
        output_path (str): Path where the output file will be saved
    """
    # Read both CSV files
    dataset = pd.read_csv(dataset_path)
    main_df = pd.read_csv(main_path)
    
    # Verify scalar_text_sequentiality exists in main_df
    if 'scalar_text_sequentiality' not in main_df.columns:
        raise ValueError("Column 'scalar_text_sequentiality' not found in main.csv")
    
    # Check if required columns exist in dataset
    required_columns = ['recAgnPairId', 'recImgPairId', 'AssignmentId']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    
    # Merge the dataframes on AssignmentID
    merged_df = dataset.merge(main_df[['AssignmentId', 'scalar_text_sequentiality']], 
                            on='AssignmentId', 
                            how='inner')
    
    # Apply the determine_bin function to categorize stories
    merged_df['story_type'] = merged_df.apply(determine_bin, axis=1)
    
    # Initialize empty DataFrame for selected stories
    selected_stories = pd.DataFrame()
    
    # Process each story type
    for story_type in ['Recall', 'Retold', 'Imagined']:
        # Get stories of current type
        type_stories = merged_df[merged_df['story_type'] == story_type]
        
        print(f"\nProcessing {story_type} stories:")
        print(f"Total available: {len(type_stories)}")
        
        if len(type_stories) == 0:
            continue
            
        # Calculate mean and standard deviation
        mean_seq = type_stories['scalar_text_sequentiality'].mean()
        std_seq = type_stories['scalar_text_sequentiality'].std()
        
        print(f"Mean sequentiality: {mean_seq:.4f}")
        print(f"Standard deviation: {std_seq:.4f}")
        
        # Calculate z-scores for each story
        z_scores = (type_stories['scalar_text_sequentiality'] - mean_seq) / std_seq
        
        # Calculate probability weights using normal distribution PDF
        # Using the PDF of the normal distribution gives higher weights to points closer to mean
        weights = norm.pdf(z_scores)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Sample stories based on weights
        n_samples = min(120, len(type_stories))
        selected_indices = np.random.choice(
            type_stories.index, 
            size=n_samples, 
            replace=False,  # Sample without replacement
            p=weights
        )
        
        selected = type_stories.loc[selected_indices]
        selected_stories = pd.concat([selected_stories, selected])
        
        # Calculate and print statistics about selected stories
        selected_mean = selected['scalar_text_sequentiality'].mean()
        selected_std = selected['scalar_text_sequentiality'].std()
        
        print(f"\nSelected {n_samples} stories")
        print(f"Selected stories mean: {selected_mean:.4f}")
        print(f"Selected stories std: {selected_std:.4f}")
        print(f"Range: {selected['scalar_text_sequentiality'].min():.4f} to {selected['scalar_text_sequentiality'].max():.4f}")
        
        # Calculate percentiles of selected stories
        percentiles = [0, 25, 50, 75, 100]
        print("\nPercentiles of selected stories:")
        for p in percentiles:
            value = np.percentile(selected['scalar_text_sequentiality'], p)
            print(f"{p}th percentile: {value:.4f}")
    
    # Write selected stories to new CSV file
    output_file = output_path + 'hcV3-stories-mini.csv' if output_path.endswith('/') else output_path
    selected_stories.to_csv(output_file, index=False)
    
    print(f"\nSelected stories written to {output_file}")
    print(f"Stories per type:")
    print(selected_stories['story_type'].value_counts())

def run_sequential(recall_length:int):
    """
    Function that runs the entire model in one process rather than split between models
    """
    data = pd.read_csv("./datasets/hcV3-stories-760.csv")
    
    # df for writing
    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                        "scalar_text_sequentiality",
                                        "sentence_total_sequentialities",
                                        "sentence_contextual_sequentialities",
                                        "sentence_topic_sequentialities",
                                        "story",
                                        "recAgnPairId",
                                        "recImgPairId"])

    # load model once
    model = SequentialityModel("neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8",
                            topic="A short story",
                            recall_length=recall_length)
    
    # load the tokens into the cache
    model.load_tokens_to_cache("./datasets/hcv3-stories-tokens.csv")
    

    times = []

    data_size = 760
    for i in range(data_size):
        vec = data.iloc[i]

        start_time = time.perf_counter()
        seq = model.calculate_text_sequentiality(vec.story)
        sequentialities.loc[len(sequentialities)] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

        compute_time = time.perf_counter() - start_time
        times.append(compute_time)
        
        if (i+1) % 10 == 0:
            with open(f"./outputs/llama-70b-quantized/{recall_length}/log.txt", "w") as file:
                file.write(f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}")

    print(f"total time to complete: {np.sum(times):.4f}")

    sequentialities.to_csv(f"./outputs/llama-70b-quantized/{recall_length}/main.csv")


def test_bed():
    """
    Function to test whether parallel or sequential running is faster
    """

    data = pd.read_csv("./datasets/hcV3-stories-mini.csv")
    tmp = []

    for x in [5, 10, 15, 20, 25]:
        start_time = time.perf_counter()
        model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct",
                                topic="A short story",
                                recall_length=4)

        model.load_tokens_to_cache("./datasets/hcv3-stories-tokens.csv")
        
        for i, row in enumerate(data.iterrows()):
            seq = model.calculate_text_sequentiality(row[1]["story"])

            # save it - assume it's constant time

            print(f"row: {i} seq: {seq[0]}")

            if i == (x-1):
                break
        t = time.perf_counter() - start_time
        print(f"sequential time: {t}")

        tmp.append(t)

        del model
    
    print(tmp)

    # del model

    # # parallel - pseudo: 
    # start_time = time.perf_counter()

    # for i, row in enumerate(data.iterrows()):
    #     model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct",
    #                         topic="A short story",
    #                         recall_length=4)
        
    #     seq = model.calculate_text_sequentiality(row[1]["story"])

    #     # save it - assume it's constant time

    #     print(f"row: {i} seq: {seq[0]}")

    #     if i == 4:
    #         break

    #     del model
    
    # print(f"parallel time: {time.perf_counter() - start_time}")


import pandas as pd
import re
import os
import torch
from tqdm import tqdm
import json

def preprocess_dataset(csv_path, model_class, model_params):
    """
    Tokenize the entire dataset once and save to a new CSV.
    
    Args:
        csv_path: Path to the input CSV file
        model_class: The SequentialityModel class
        model_params: Dictionary of parameters to initialize the model
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the dataset
    print(f"Reading dataset from {csv_path}")
    data = pd.read_csv(csv_path)
    
    # Initialize model for tokenization
    print("Initializing tokenization model...")
    model = model_class(**model_params)
    
    # Prepare dataframe for tokenized data
    tokenized_data = pd.DataFrame()
    tokenized_data['AssignmentId'] = data['AssignmentId']
    tokenized_data['recAgnPairId'] = data['recAgnPairId']
    tokenized_data['recImgPairId'] = data['recImgPairId']
    tokenized_data['story'] = data['story']
    
    # Process each story
    print("Processing stories...")
    tokenized_sentences_list = []
    
    for i in tqdm(range(len(data))):
        story = data.iloc[i].story
        sentences = re.split(r'(?<!\.\.\.)[\.\?\!](?!\.)\s*', story)
        
        # Clean and pair sentences
        processed_sentences = []
        for j in range(0, len(sentences) - 1, 2):
            if j+1 < len(sentences):
                sentence = sentences[j].strip() + sentences[j + 1]
                processed_sentences.append(sentence)
        
        # Tokenize all sentences in the story
        tokenized_sentences = []
        for sentence in processed_sentences:
            tokens = model._tokenize_with_cache(sentence)
            # Convert tensor to list if necessary
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            tokenized_sentences.append(tokens)
        
        # Store the tokenized sentences
        tokenized_sentences_list.append(json.dumps(tokenized_sentences))
    
    # Add tokenized sentences to the dataframe
    tokenized_data['tokenized_sentences'] = tokenized_sentences_list
    
    # Write to new CSV
    output_path = os.path.join(output_dir, 'hcv3-stories-tokens.csv')
    print(f"Writing tokenized data to {output_path}")
    tokenized_data.to_csv(output_path, index=False)
    
    print("Preprocessing complete!")
    return output_path

# Example usage:
if __name__ == "__main__":
    # model_params = {
    #     "model_name": "neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8", 
    #     "topic": "A short story",
    #     "recall_length": 4
    # }
    
    # preprocess_dataset(
    #     csv_path="./datasets/hcV3-stories-760.csv",
    #     model_class=SequentialityModel,
    #     model_params=model_params
    # )

    # this is how it was run on hpc3 - function is in verification/verify_seq.py
    # verify_data(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    # this is the equivalent of verify_data but run sequentially rather than parallel
    run_sequential(int(sys.argv[1]))


    # df_760 = make_proportional_subset_using_other_subset(data="./datasets/hcV3-stories.csv", other="./datasets/hcV3-stories-mini-filtered.csv")

    # df_760.to_csv("./datasets/hcV3-stories-760.csv")

    # df = pd.read_csv("./datasets/hcV3-stories-mini-filtered.csv")

    # df_1000 = pd.concat([df, df_760])

    # df_1000.to_csv("./datasets/hcV3-stories-1000.csv")

    # test_bed()

    # x = [5, 10, 15, 20, 25]
    # s1 = [51.877457665978, 69.82546812505461, 102.48368912516162, 132.18100025015883, 161.17464420897886]
    # s2 = [44.23506679106504, 77.46961391717196, 111.98336641700007, 154.9341577081941, 160.76373516698368]
    # s3 = [43.1074358751066, 73.00900541618466, 100.92742945882492, 133.93042233306915, 163.73520620795898]
    # s4 = [44.16905250004493, 68.6806848749984, 99.41008437494747, 127.99263358395547, 161.46826220816]

    # plt.figure()
    # plt.plot(x, s1, label="s1", color="blue")
    # plt.plot(x, s2, label="s2", color="purple")
    # plt.plot(x, s3, label="s3", color="green")
    # plt.plot(x, s4, label="s4", color="red")
    # # plt.plot(x, [percentage_dif(s1, s2) for s1, s2 in zip(s, p)])
    # # plt.ylim(top=150, bottom=0)
    # # plt.plot(x, s1, label="sequential w/ optimization", color="red")
    # plt.legend()
    # plt.show()


    # Find values near mean and save result
    # select_stories_near_mean("./data/hcV3-stories.csv", "./data/calculated_values/4/main.csv", "./data/")
    # create_mini_files()

    # generate plots
    # generate_data_proportion_chart(file_path="./datasets/hcV3-stories.csv", title="Proportions of hcV3-stories.csv")
    # generate_data_proportion_chart(file_path="./datasets/hcV3-stories-1000.csv", title="Proportions of hcV3-stories-1000.csv")
    # generate_plots(file_name = "main-mini.csv")
    
