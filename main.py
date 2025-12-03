from verification.generate_plots import generate_2d, generate_2a, create_balanced_dataset, generate_data_proportion_chart, percentage_dif
from verification.verify_seq import *
from verification.subset import analyze_embeddings, save_top_stories, merge_top_stories, determine_bin, make_large_subset, make_proportional_subset_using_other_subset
from src.embedding import SequentialityEmbeddingModel # this is the USE model
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import torch
import gc
import os


# Models:
"microsoft/Phi-3-mini-4k-instruct"
"SakanaAI/TinySwallow-1.5B-Instruct"
"meta-llama/Llama-3.3-70B-Instruct"
"neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8"
"meta-llama/Llama-3.2-3B-Instruct"

# non-prompt finetuned
"openai-community/gpt2-xl"
"allenai/OLMo-2-1124-13B"


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


# Example usage:
if __name__ == "__main__":
    # This is the function to use when running on hpc - see documentation for parameters
    # run_sequential(int(sys.argv[1]))
    pass