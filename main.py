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
"microsoft/Phi-3-mini-4k-instruct"
"SakanaAI/TinySwallow-1.5B-Instruct"
"meta-llama/Llama-3.3-70B-Instruct"
"neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8"
"meta-llama/Llama-3.2-3B-Instruct"


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
    save_path = "./outputs/llama-3b/"  # CHANGE THIS

    data = pd.read_csv("./datasets/hcV3-stories.csv")
    
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
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct",  # CHANGE THIS
                            topic="A short story",
                            recall_length=recall_length)

    times = []

    data_size = 6854   # CHANGE THIS
    for i in range(data_size):
        try: # try running the model
            vec = data.iloc[i]

            story = vec.story
            topic = vec.mainEvent

            start_time = time.perf_counter()
            seq = model.calculate_text_sequentiality(story, topic=topic)
            sequentialities.loc[len(sequentialities)] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

            compute_time = time.perf_counter() - start_time
            times.append(compute_time)
            print((f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}"))
            
            if (i+1) % 10 == 0:
                with open(f"{save_path}{recall_length}/log.txt", "w") as file:
                    file.write(f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}")
        
        except Exception as e: # dump sequentialities into a file even if it errors out
            sequentialities.to_csv(f"{save_path}{recall_length}/main.csv")
            print(e)

            quit(-42)

    print(f"total time to complete: {np.sum(times):.4f}")

    sequentialities.to_csv(f"{save_path}{recall_length}/main.csv")

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

def run_film_fest_init(stories:list[str]):
    """
    Function that calculates the value for this story and writes it to a csv
    """
     # load model once
    model = SequentialityModel("neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8",  # CHANGE THIS
                            topic="A short story",
                            recall_length=9)
    
    sequentialities = []
    for i in range(len(stories)):
        text = " ".join(stories[:i])

        seq = model.calculate_text_sequentiality(text)

        sequentialities.append(seq[0])

        if i == len(stories) - 1:  # print the sequentialities for every sentence in the entire story
            print(seq[1])
    
    print(f"sequentialities by scene: {sequentialities}")

    
    

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
    
    # stories = [""""THE BOYFRIEND" in bold white text fades in on a black screen before fading out. The letters of "high maintenance" appear in the center of the screen one by one in white text. A simple jingle plays in the background.""",
    #         """View of the back of a man's head. He's sitting at a table across from a woman and eating dinner. The woman looks at the man. View of the man over the woman's shoulder. He's looking down at his plate and cutting something with his knife and fork. Close up of the woman's face. She's looking down and chewing something. Close up of the man's face. He's also looking down and chewing. Side view of the woman's face. She uses her fork to eat something before looking up. Side view of the man's face. He eats a bite while looking down.""",
    #         """The woman stands up and grabs a bottle of wine to her side. She walks over to the man. The woman begins to pour wine for the man, but he covers his glass with his hand. The man, without looking up says "not for me dear" and continues eating. The woman sits back down and glances at the man before saying "you know it's our anniversary".""",
    #         """The man stops eating and looks at the woman, saying "you know I can't drink wine or alcohol" before starting to eat again. The woman picks up her wine glass and downs it in a few gulps while looking at the man. When she finishes, she sighs and looks at the man pointedly. The woman sets down her wine glass with a "clink" sound. She smiles, sighs and then pours more wine for herself. The man says "the asparagus is very tender".""",
    #         """The woman replies "oh yeah". The man continues to eat without looking up. Side view of the woman's face. She says "They say it's an aphrodisiac" while looking at the man. Side view of the man's face. He looks up and asks "Who does?" The woman replies "Oh, I don't know, people. That's just what people say" while eating another bite of food. The man continues to look down and not say anything. The woman glances up at the man but doesn't say anything.""",
    #         """The man looks up and says "So dear, how was your day?" The woman, furious, gets up and says "Oh geez, shut up and have a drink!" as she throws her glass of wine at the man. The woman sits down, agitated and says "You're a computer analyst, not a fucking surgeon! Why can't you, I don't know, relax for once in your life!" """, 
    #         """The man takes something out of his pocket. The woman says angrily "Oh don't you dare". The man, with wine still on his face puts a cigarette in his mouth and starts to smoke. The woman stands up and says "Don't you dare smoke in my house!" The man starts to smoke but the woman storms over and pulls the cigarette out of his mouth. The woman walks back across the room and angrily puts the cigarette out in an ashtray. The woman looks back across the room to see that the man is lighting another cigarette.""",
    #         """The woman sits down and says, frustrated, "Yeah, that's right, passive smoke. That's exactly what I wanted for my anniversary". The man continues to look down and smoke. The woman continues, "followed by some stilted conversation. And if I'm really lucky, by some short, mechanical sex." The man looks up at the woman. The woman looks down, avoiding eye contact for a moment but glances up. The man continues to stare at the woman. He swallows. The woman looks back before saying…""",
    #         """The woman says "I'm sorry." and begins to stand up. The woman walks across the room to the man, and says "Come here". The woman sits down in the man's lap and hugs him, saying "There, there. I didn't mean what I said". The woman holds the man's head closer, stroking his hair.""", 
    #         """The woman's hand traces the man's head to his neck. She pulls down the man's shirt collar to reveal a switch on his neck. Close up of the woman's face. She blinks. The woman's hand slides to the switch. She pushes it down with a "click" sound. Close up of the man's face. With a whine sound, his neck droops and he lies on the woman's shoulder, seemingly unconscious.""", 
    #         """The woman pushes the man off her shoulder with a grunt, stands up, and walks away. His head hangs there, limp. The woman sits down at the other side of the table. She takes off her shoes and crosses her legs on the table, sighing. She says sarcastically "Happy anniversary".""", 
    #         """Close up of the back of a laptop. The camera pans up to the woman, typing something. Metallophone music starts playing in the background. On her computer screen: "PROMETEUS ROBOTICS CHOOSE EVERLASTING LOVE". With a click, the screen displays a lineup of six differently dressed men. The woman types something before looking up. Across the room, the man still sits there, his head down, hanging limply. The woman looks back down at her screen before smiling. On her screen: several pictures of a muscular, dark haired man.""",
    #         """The woman puts on a bluetooth headset before looking down at her screen again. Screen briefly shows the same page with the man, before switching to a screen that says "...DIALING: 0800 800 800..." Sounds of a phone being dialed in the background. Close up of the woman's face. A woman on the other end of the phone says "Hello you've reached Rachelite technical support, how may I help you?" The woman replies: "Hi. I, I'm unsatisfied with my current unit". The customer support rep says "Okay, what model do you currently have?" The camera pans from the computer screen to a picture frame of the woman in a wedding dress hugging the man. Woman: "It's the 100 series" Customer service: "And what seems to be the problem?" Close up of the woman's face. Woman: "He lacks ambition. He has no sense of adventure." Customer service representative: "Yes, that is a common malfunction with the 100 series.""",
    #         """The customer service rep continues, "Would it help if we upgraded you to a higher model?" Woman: "Yeah, would you be able to..." Camera cuts to a view of the man sitting there, still limp. The woman continues… "give me something". Camera cuts back to a view of the woman. She continues, "a bit sportier?" Customer service rep: "Certainly. Do you have any preferences?" Woman: "Oh, um…." as she clicks the computer.""",
    #         """View of the computer screen, showing the same man. Woman: "A rock climber. Oh no, a masseuse". Camera slowly pans away from the woman at her computer. Woman: "Oh wait. A rock climbing masseuse? Yeah, like the picture. Yeah, but no beard. Maybe just a 5 o'clock shadow. Yeah. And, could he have shorter hair? And blonde. Yeah. And…." The camera fades to black.""",
    #         """A loud banging sound on the black screen. Sudden cut to a view of a doorbell. The woman's eye approaches the doorbell.""",
    #         """A clicking sound as the door unlocks. The woman opens up the door to a delivery girl looking down at her clipboard. A large box that says "Prometeus Robotics…." is pushed into frame, and another delivery girl steps into frame. Someone walks across frame to reveal a  layer of plastic. Sound of wrinkling plastic as it is pulled away. Close up of the woman's face. She blinks, and the sound of wrinkling plastic continues. The delivery woman peels back the plastic to reveal a man, exactly as the woman described earlier. Camera slowly zooms in on the woman's face. She looks intensely ahead. A delivery woman walks behind her. String music swells in the background.""", 
    #         """The woman turns her head to see the delivery women have tied the original man to a cart and start to wheel him away. Camera pans out from the woman's face as she looks at the original man. The new man is in the background, still partially wrapped in plastic. The woman starts walking over to the original man and the delivery women and says "Um wait please". The woman gives the original man a quick kiss before backing away.""",
    #         """Close up of the woman's hands as she takes a ring off of the original man's ring finger.""",
    #         """Close up of the woman's hands as she puts the ring on the new man's ring finger. She sets the man's hand down and pats it gently.""",
    #         """Close up of the woman's hands on the man's neck. She flicks the switch with a click, and an electronic whine starts. The whine continues. The woman walks across the frame and the man sits there motionless. The woman sits down and looks expectantly at the man. The man blinks. With an electronic whine, he looks around. He then grabs the napkin next to him and sits up.""",
    #         """The woman asks cautiously "Glass of wine?" and looks at the man. The man, looking down, replies "No thanks, I've got a big climb tomorrow as he eats a bite of food. The woman looks back before looking down again.""",
    #         """The man looks at the woman and says "the asparagus is very tender". The woman looks back hesitantly before smiling, nodding and saying "Yes, dear". Back to the man, who says "They say it's an aphrodisiac, you know?" while eating a piece. The woman, smiling, asks "Who does?" The man, shaking his head and smiling slightly says "I don't know, it's just what people say" before standing up.""",
    #         """The man slowly walks across the room until he's behind the woman.""",
    #         """The man starts to massage the woman's shoulders. He says, "So, tell me about your day dear". The woman says "Oh, I went into town this morning to pick up some stuff for tonight." The man looks down while continuing to massage her. The woman continues, "then um, I went for light lunch with Anna". Close up of the man's hands. He says "Hm? Then what?" Back to the woman, who, chuckling, says "Um, and then I decided…" Close up of the man's hands. His hands move up her shoulders. The woman continues, "since it's our anniversary, too." The man's left hand moves up to the woman's neck. Woman: "...treat myself and I..." """,
    #         """With a click sound, the woman suddenly stops talking. The woman's neck goes limp and her head hangs. The man looks up and smiles slightly before walking away. View of the woman's face hanging there, expressionless. The man walks away""",
    #         """View of the woman's neck, revealing a switch. Camera pans up. In the background the man walks across the room and sits down on a sofa with a cigarette and turns on the TV, crossing his legs. Close up of the man's emotionless face as he smokes, with the TV going in the background.""",
    #         ]


    # run_film_fest_init(stories)

    # create_mini_files(base_path="./outputs/llama-3b", merged_file="./datasets/hcV3-stories-quartered.csv")
    # generate_plots(data_path="./outputs/llama-70b-quantized", file_name="main.csv")

    # generate plots
    # generate_data_proportion_chart(file_path="./datasets/hcV3-stories.csv", title="Proportions of hcV3-stories.csv")
    # generate_data_proportion_chart(file_path="./datasets/hcV3-stories-quartered.csv", title="Proportions of hcV3-stories-quartered.csv")
    # generate_plots(data_path="./outputs/llama-3b/", file_name = "main.csv")
    
