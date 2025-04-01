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
    data = pd.read_csv("./datasets/hcV3-stories-quartered.csv")
    
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
    model = SequentialityModel("neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8",  # CHANGE THIS
                            topic="A short story",
                            recall_length=recall_length)

    times = []

    data_size = 1713   # CHANGE THIS
    for i in range(data_size):
        try: # try running the model
            vec = data.iloc[i]

            start_time = time.perf_counter()
            seq = model.calculate_text_sequentiality(vec.story)
            sequentialities.loc[len(sequentialities)] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

            compute_time = time.perf_counter() - start_time
            times.append(compute_time)
            print((f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}"))
            
            if (i+1) % 10 == 0:
                with open(f"./outputs/llama-70b-quantized/{recall_length}/log.txt", "w") as file:
                    file.write(f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}")
        
        except Exception as e: # dump sequentialities into a file even if it errors out
            sequentialities.to_csv(f"./outputs/llama-70b-quantized/{recall_length}/main.csv")
            print(e)

            quit(-42)

    print(f"total time to complete: {np.sum(times):.4f}")

    sequentialities.to_csv(f"./outputs/llama-70b-quantized/{recall_length}/main.csv")  # CHANGE THIS

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


def run_film_fest_init(stories:list[str]):
    """
    Function that calculates the value for this story and writes it to a csv
    """
    recall = 9
    print("loading model...")
    # load model once
    model = SequentialityModel("SakanaAI/TinySwallow-1.5B-Instruct",
                            topic="A description of a short film",
                            recall_length=recall)
    
    print("model loaded!\nstarting calculation")
    for story in stories:
        total, sentence, contextual, topic = model.calculate_text_sequentiality(story)

        print("sequentiality for each sentence in the story:")
        print(sentence)

    del model #clean up

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
        
        sentences = re.findall(r'([^.!?]+[.!?]+)', story)

        sentences = [s.strip() for s in sentences if s.strip()]
        
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
    # Code to preprocess tokens
    # model_params = {
    #     "model_name": "neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8", 
    #     "topic": "A short story",
    #     "recall_length": 4
    # }
    
    # preprocess_dataset(
    #     csv_path="./datasets/hcV3-stories-quartered.csv",
    #     model_class=SequentialityModel,
    #     model_params=model_params
    # )

    # this is how it was run on hpc3 - function is in verification/verify_seq.py
    # verify_data(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    # this is the equivalent of verify_data but run sequentially rather than parallel
    run_sequential(int(sys.argv[1]))
    
    # stories = [
    #         """
    #         This is a stress test? I really hope this works... A.G. made this. Dr. Su wants to make sure that it works too! "help me?" he said.
    #         """,
    #         """
    #         Elias found the broken pocket watch buried in the sand just outside the abandoned library. It was stuck at 3:47, the exact time his grandfather had vanished decades ago. He ran his thumb over the cracked glass, wondering if this was a coincidence—or a message. Inside the library, dust coated the shelves, but the checkout desk held a single, faded library card. The name had been smudged beyond recognition, but the last book borrowed stood out: The Vanishing Hour. Elias pulled it from the shelf and flipped through the pages. A red feather slipped out, landing on the floor. The tip was burned. He shivered. His grandfather had once told him a story about birds that carried fire in their wings. As he placed the feather back in the book, something rattled inside its spine. He carefully pried it open and found a brass key with no teeth. A key that opened nothing. Or maybe… something that didn’t need a lock. His hand instinctively went to his pocket, where he kept a peculiar glass marble filled with swirling silver mist. It had been his grandfather’s, a keepsake he never understood. But now, holding the key in one hand and the marble in the other, he heard it—whispers, faint and distant. He closed his eyes, listening. The words became clearer: “Come find me. Before time runs out.” The watch ticked once. Then stopped. Elias exhaled. The hunt was on.
    #         """,
    #         """
    #         The train station was empty when Mara arrived, save for a single torn ticket caught beneath a bench. She picked it up. Only the date was visible—today’s. The destination had been ripped away. Something rattled in her bag. She hesitated before pulling out the locked wooden box. It had belonged to her mother, passed down from a grandmother she had never met. She’d never found the key, but tonight, it felt heavier, as if waiting for something. A whisper of wind made her turn. On the station wall, taped to the glass, was a child’s drawing of a house. The shape was familiar—her grandmother’s old home—but the windows were shaded black, as if hiding something inside. Mara’s grip tightened around the box. Her mother had warned her never to visit that house at night. Never to enter. But she had never said why. A flicker caught her eye—a rusted lantern with no wick, left abandoned near the tracks. She picked it up, and through the green-tinted glass, she saw something glint inside the base. She pried it open and fished out a silver spoon, its handle engraved: J.L. Her grandmother’s initials. The train whistle blew in the distance. She looked at the torn ticket. Then at the lantern. Then at the box. She had a choice to make. And she knew, deep down, that whatever she chose… the house was waiting.
    #         """
    #     ]
    # run_film_fest_init(stories)


    # x = list(range(1, 132))
    # seq = [0.013144356863839286, 1.2882036481584822, 0.8223353794642857, 1.2403020858764648, 1.663923375746783, 0.9185791015625, 1.0516199747721353, 2.01287841796875, 1.2958984375, 3.5830590384347096, 1.17109375, 2.984400885445731, 1.299609375, 3.3293079024269465, 1.637407938639323, 0.924891471862793, 2.9759114583333335, 3.0057915581597223, 4.1131439208984375, 2.9813819298377404, 2.687631607055664, 1.6726830523947012, 1.7950349287553267, 2.656444549560547, 1.4207801818847656, 2.698883056640625, 2.178282304243608, 3.0744384765625, 0.45610809326171875, 2.7419921875, 2.71431884765625, 1.2265625, 1.1523818969726562, 1.8839754377092635, 0, 4.552083333333333, 1.536279296875, 1.23732421875, 3.03857421875, 3.0965169270833335, 2.8037923177083335, 1.0764973958333333, 1.6930803571428572, 0, 1.6593068440755208, 0.4886474609375, 0.5812759399414062, 0, 1.0833296342329546, 2.8576178550720215, 2.899222903781467, 3.4982705646091037, 1.680597686767578, 1.3046875, 3.861837213689631, 1.58447265625, 1.0649573284646738, 1.1930338541666667, 3.363986545138889, 1.6702714399857954, 1.1769211713005514, 3.491030693054199, 1.1409912109375, 4.417874654134114, 1.7004720052083333, 2.14697265625, 1.3665823936462402, 1.7880859375, 0.7770787752591647, 1.1415182260366588, 1.2298828125, 1.2107514880952381, 0.8391510009765625, 1.321417864631204, 0, 2.6124343872070312, 2.241650390625, 2.0873046875, 0.8433868408203125, 0, 0.890771230061849, 1.9272904829545454, 2.496875, 1.3914930555555556, 0, 0.7452256944444444, 2.7682291666666665, 4.12548828125, 4.453125, 2.0037109375, 1.6102362738715277, 4.07568359375, 3.3489583333333335, 2.5260416666666665, 2.010532924107143, 2.494537353515625, 2.5482421875, 4.051215277777778, 0.9527413050333658, 5.024881998697917, 3.4247349330357144, 1.9347076416015625, 1.6064506199048914, 2.7578887939453125, 0.665418122944079, 2.7162000868055554, 3.2390625, 0.7458943684895833, 2.8917410714285716, 1.27646484375, 2.623046875, 2.6168118990384617, 2.4189453125, 0.9529880947536893, 1.293701171875, 1.3093109130859375, 2.4090169270833335, 4.422119140625, 1.8671875, 4.347464425223214, 1.7135881696428572, 2.91103515625, 2.1979166666666665, 2.53216552734375, 1.1111111111111112, 2.3277994791666665, -0.17281150817871094, 2.431361607142857, 2.896875, 1.8603891225961537, 1.5613839285714286]
    # seq1 = [-0.020670572916666668, 1.2595011393229167, 1.181978013780382, 1.200726318359375, 2.10980332439596, 1.2406782670454546, 1.774267578125, 2.400390625, 2.1338704427083335, 0.7711458206176758, 2.056734561920166, 1.425800051007952, 3.911328125, 1.8787109375, 0.6674681163969494, 2.1182774030245266, 1.350280211522029, 1.7837858200073242, 3.2620985243055554, 1.189727783203125, 0.9255625406901041, 5.66845703125, 3.393146514892578, 1.876708984375]
    # seq2 = [0.08878366570723684, 1.9322338104248047, 2.927734375, 4.225667317708333, 1.34381103515625, 1.0041717529296874, 0.8162550926208496, 1.6382606907894737, 2.755580357142857, 0.5838894314236112, 1.3565118963068181, 2.8967459542410716, 0.6288967132568359, 3.51171875, 2.1002400716145835, 1.2196758270263672, 0.674701603976163, 1.0997890896267362, 5.51953125, 6.118408203125, 1.6428309849330358, -0.9730631510416666, 5.604705810546875, 3.481109619140625, 0.9888814290364584, 0.9737091064453125]
    
    # plt.figure()
    # plt.plot([x for x in range(len(seq2))], seq2)
    # plt.xlabel("sentences")
    # plt.ylabel("seqentiality")
    # plt.show()


    # create_mini_files(merged_file="./datasets/hcV3-stories-quartered.csv")
    # generate_plots(data_path="./outputs/llama-70b-quantized", file_name="main.csv")

    # generate plots
    # generate_data_proportion_chart(file_path="./datasets/hcV3-stories.csv", title="Proportions of hcV3-stories.csv")
    # generate_data_proportion_chart(file_path="./datasets/hcV3-stories-quartered.csv", title="Proportions of hcV3-stories-quartered.csv")
    # generate_plots(data_path="./outputs/llama-70b-quantized/", file_name = "main.csv")
    
