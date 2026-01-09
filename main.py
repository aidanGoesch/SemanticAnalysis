from verification.generate_plots import generate_2d, generate_2a, create_balanced_dataset, generate_data_proportion_chart, percentage_dif
from verification.subset import analyze_embeddings, save_top_stories, merge_top_stories, determine_bin, make_large_subset, make_proportional_subset_using_other_subset
# from src.embedding import SequentialityEmbeddingModel # this is the USE model
from src.sequentiality import calculate_sequentiality, calculate_sequentiality_statistics, SequentialityModel
import pandas as pd
# import tensorflow_hub as hub
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import torch
import gc
import sys
import os
import time
import numpy as np


# Models:
"microsoft/Phi-3-mini-4k-instruct"
"SakanaAI/TinySwallow-1.5B-Instruct"
"meta-llama/Llama-3.3-70B-Instruct"
"neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8"
"meta-llama/Llama-3.2-3B-Instruct"

# non-prompt finetuned
"openai-community/gpt2-xl"
"allenai/OLMo-2-1124-13B"

# models to test
MODEL_IDS = ["SakanaAI/TinySwallow-1.5B-Instruct",
            "neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8",
            "openai-community/gpt2-xl",
            "allenai/OLMo-2-1124-13B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct"]


# HPC Checklist
#   - Is the model name correct?
#   - Is the save file location correct?
#   - Is the dataset file path correct?
#   - Is the correct function being run with the correct arguments in main.py?
#   - Is the version of the code on HPC what you want to run?


def generate_plots(data_path:str="./outputs/phi-4k-mini", file_name:str="main.csv", model_name:str="model"):
    """
    Function that generates the graph from the calculated sequentiality values. 
    Takes as an argument the path to where the data is stored, and the filename
    """

    dfs = []
    for i in range(9):
        dfs.append(pd.read_csv(f"{data_path}/{i + 1}/{file_name}"))

    generate_2a(dfs, model_name)
    generate_2d(dfs, model_name)


def run_sequential(recall_length:int):
    """
    Function that runs the entire model in one process rather than split between models. Currently
    runs HIPPOCORPUS data.
    USE THIS AS A TEMPLATE FOR RUNNING SEQUENTIALITY ON DIFFERENT DATA
    """
    save_path = "./outputs/embedding/"  # CHANGE THIS

    data = pd.read_csv("./datasets/hcV3-stories.csv")
    
    # df for writing
    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                        "scalar_text_sequentiality",
                                        "sentence_total_sequentialities",
                                        "sentence_contextual_sequentialities",
                                        "sentence_topic_sequentialities",
                                        "story",
                                        "recAgnPairId",
                                        "recImgPairId",
                                        "memType"])

    # load model once
    model = SequentialityModel(model_name="CHANGE THIS",
                               topic="CHANGE THIS",
                               recall_length=0)

    times = []

    recall_str = str(recall_length)
    # Construct the full directory path
    dir_path = os.path.join(save_path, recall_str)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    data_size = 6854   # CHANGE THIS
    for i in range(data_size):
        try: # try running the model
            vec = data.iloc[i]

            story = vec.story
            topic = vec.mainEvent

            start_time = time.perf_counter()
            seq = model.calculate_text_sequentiality(story, topic=topic)
            sequentialities.loc[len(sequentialities)] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId] + [vec.memType]

            compute_time = time.perf_counter() - start_time
            times.append(compute_time)
            print((f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}"))
            
            if (i+1) % 10 == 0:
                with open(f"{save_path}{recall_length}/log.txt", "w") as file:
                    file.write(f"iteration ({i+1}/{data_size}) sequentiality value: {seq[0]:.4f}     time to complete: {compute_time:.4f}     time elapsed: {np.sum(times):.4f}     time remaining: ~{np.mean(times) * (data_size - i - 1):.4f}")
        
        except Exception as e: # dump sequentialities into a file even if it errors out
            sequentialities.to_csv(f"{save_path}{recall_length}/240.csv")
            print(e)

            quit(-42)

    print(f"total time to complete: {np.sum(times):.4f}")

    sequentialities.to_csv(f"{save_path}{recall_length}/full.csv")


def get_annotations() -> list[str]:
    """
    Function that returns the list of annotations so that it looks prettier in the following function
    """
    sentences = [
    "M and K do a crossword puzzle together.",
    "M suggests adopting a dog; K isn't receptive.",
    "M is reprimanded unfairly by her boss.",
    "K goes to the dentist.",
    "K is bored at dinner with M's parents.",
    "M bumps into an old friend who works at a top law firm.",
    "K misses M's piano recital.",
    "M is offered a high-paying job in another state.",
    "K buys M a birthday present.",
    "M sees that K forgot to take out the trash.",
    "M confronts K and demands a divorce."
]
    return sentences


def generate_model_sequentiality(model_idx:int):
    """
    Function that is run on HPC to test a specific model from the model id 
    """
    if model_idx not in range(len(MODEL_IDS)):
        print("model id out of bounds")
        return
    
    model_name = MODEL_IDS[model_idx]
    print(f"\n{'='*60}")
    print(f"Starting sequentiality generation for model {model_idx}: {model_name}")
    print(f"{'='*60}\n")
    
    # Clear corrupted cache
    import shutil
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe_cache_name = model_name.replace('/', '--')
    
    print(f"Checking for cached model files: {safe_cache_name}")
    model_caches = list(cache_dir.glob(f"*{safe_cache_name}*"))
    
    if model_caches:
        print(f"Found {len(model_caches)} cached entries. Clearing...")
        for cache in model_caches:
            try:
                print(f"  Removing: {cache}")
                shutil.rmtree(cache, ignore_errors=True)
            except Exception as e:
                print(f"  Warning: Could not remove {cache}: {e}")
        print("Cache cleared successfully")
    else:
        print("No cached entries found")
    
    print("\nInitializing SequentialityModel...")
    try:
        model = SequentialityModel(
            model_name=model_name, 
            topic="something",
            recall_length=9
        )
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to initialize model")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nModel loaded successfully. Starting sequentiality calculations...")
    
    # read in filmfest data
    scenes = get_annotations()

    output = pd.DataFrame(columns=["scalar_text_sequentiality",
                            "sentence_total_sequentialities",
                            "sentence_contextual_sequentialities",
                            "sentence_topic_sequentialities",
                            "topic"])

    # calculate sequentiality
    # no divorce topic
    print("\nCalculating sequentiality for topic 1 (no divorce)...")
    topic_1 = "A short story about M and K"
    total_text_sequentiality, sentence_level_sequentiality, contextual_sentence_level_sequentiality, topic_sentence_level_sequentiality = model.calculate_text_sequentiality(" ".join(scenes), topic=topic_1)

    # save to file
    output.loc[len(output)] = [total_text_sequentiality,
                                sentence_level_sequentiality,
                                contextual_sentence_level_sequentiality,
                                topic_sentence_level_sequentiality,
                                topic_1]
    
    print(f"Topic 1 sequentiality: {total_text_sequentiality:.4f}")
    
    # divorce topic
    print("\nCalculating sequentiality for topic 2 (divorce)...")
    topic_2 = "A short story about M and K getting a divorce"
    total_text_sequentiality, sentence_level_sequentiality, contextual_sentence_level_sequentiality, topic_sentence_level_sequentiality = model.calculate_text_sequentiality(" ".join(scenes), topic=topic_2)

    # save to file
    output.loc[len(output)] = [total_text_sequentiality,
                                sentence_level_sequentiality,
                                contextual_sentence_level_sequentiality,
                                topic_sentence_level_sequentiality,
                                topic_2]
    
    print(f"Topic 2 sequentiality: {total_text_sequentiality:.4f}")
    
    os.makedirs("./outputs/benchmarking/", exist_ok=True)
    
    # sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    output_path = f"./outputs/benchmarking/{safe_model_name}.csv"
    output.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


def run_ai_generated_stories():
    models = ["allenai/OLMo-2-1124-13B"]

    # Load datasets
    open_ai_data = pd.read_csv("./datasets/misc/syntehtic-stories-openai.csv")
    google_data = pd.read_csv("./datasets/misc/syntehtic-stories-google.csv")
    anthropic_data = pd.read_csv("./datasets/misc/syntehtic-stories-anthropic.csv")

    # Calculate sequentiality for each dataset
    open_ai_output = calculate_sequentiality(models, list(open_ai_data["story"]), list(open_ai_data["topic"]))
    google_output = calculate_sequentiality(models, list(google_data["story"]), list(google_data["topic"]))
    anthropic_output = calculate_sequentiality(models, list(anthropic_data["story"]), list(anthropic_data["topic"]))

    # Add temperature and source columns
    open_ai_output['temperature'] = open_ai_data['temperature'].values
    open_ai_output['source'] = 'openai'
    
    google_output['temperature'] = google_data['temperature'].values
    google_output['source'] = 'google'
    
    anthropic_output['temperature'] = anthropic_data['temperature'].values
    anthropic_output['source'] = 'anthropic'

    # Merge all dataframes
    merged_df = pd.concat([open_ai_output, google_output, anthropic_output], ignore_index=True)
    
    # Reorder columns for clarity
    merged_df = merged_df[['temperature', 'topic', 'source', 'scalar_text_sequentiality', 
                          'sentence_total_sequentialities', 'sentence_contextual_sequentialities', 
                          'sentence_topic_sequentialities', 'model_id']]
    
    # Save to outputs folder
    os.makedirs("./outputs/ai_generated/", exist_ok=True)
    merged_df.to_csv("./outputs/ai_generated/merged_sequentiality.csv", index=False)
    
    print(f"Merged dataframe saved with {len(merged_df)} rows")
    return merged_df



# Example usage:
if __name__ == "__main__":
    # This is the function to use when running on hpc - see documentation for parameters
    # run_sequential(int(sys.argv[1]))
    
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py <model_index>")
    #     print(f"Available models (0-{len(MODEL_IDS)-1}):")
    #     for i, model in enumerate(MODEL_IDS):
    #         print(f"  {i}: {model}")
    #     sys.exit(1)
    
    # idx = int(sys.argv[1])
    # generate_model_sequentiality(idx)
    run_ai_generated_stories()