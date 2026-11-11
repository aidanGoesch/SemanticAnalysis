"""
Calculate event segmentation sequentiality (Pieman, Monkey in the Middle, Tunnel Under the World)
for OLMo and Llama on HPC.

Same setup as the existing GPT-2 XL / Qwen / TinySwallow runs:
  - condensed stories from event_segmentation/normalized/<story>_condensed.json
  - topic "a short story", history lengths 2, 4, 6, 8
  - <model>_condensed.csv + <model>_condensed_shuffled.csv  (10 sentence + 10 word shuffles)
  - <model>_entropy.csv                                      (context / topic entropies)

Usage: python3 run_event_segmentation.py <model_index>
"""
import json
import os
import sys

from src.sequentiality import calculate_sequentiality

MODEL_IDS = ["allenai/OLMo-2-1124-13B",
             "meta-llama/Llama-3.1-8B-Instruct"]

STORIES = ["pieman", "monkey", "tunnel"]
HISTORY_LENGTHS = [2, 4, 6, 8]
N_SHUFFLE = 10

STORY_DIR = "./event_segmentation/normalized"
SAVE_DIR = "./outputs/ensemble/event_segmentation"


def load_condensed_story(story:str) -> str:
    """Join the condensed sentences in condensed_id order."""
    with open(f"{STORY_DIR}/{story}_condensed.json", "r", encoding="utf-8") as f:
        entries = json.load(f)["condensed"]
    entries = sorted(entries, key=lambda e: e["condensed_id"])
    return " ".join(entry["text"].strip() for entry in entries)


def event_segmentation(model_id:str):
    texts = [load_condensed_story(story) for story in STORIES]
    topics = ["a short story"] * len(STORIES)

    os.makedirs(SAVE_DIR, exist_ok=True)
    safe_model_name = model_id.replace("/", "_")

    # sequentiality + shuffled baseline (also writes <model>_condensed_shuffled.csv)
    calculate_sequentiality(model=model_id,
                            history_lengths=HISTORY_LENGTHS,
                            text_input=texts,
                            topics=topics,
                            n_shuffle=N_SHUFFLE,
                            save_path=f"{SAVE_DIR}/{safe_model_name}_condensed.csv")

    # entropy has to be a separate run: shuffled rows don't carry entropy columns
    calculate_sequentiality(model=model_id,
                            history_lengths=HISTORY_LENGTHS,
                            text_input=texts,
                            topics=topics,
                            calculate_entropy=True,
                            save_path=f"{SAVE_DIR}/{safe_model_name}_entropy.csv")

    print(f"Saved {safe_model_name} outputs to {SAVE_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_event_segmentation.py <model_index>")
        for i, model in enumerate(MODEL_IDS):
            print(f"  {i}: {model}")
        exit(-2)

    model_idx = int(sys.argv[1])
    if model_idx not in range(len(MODEL_IDS)):
        print("invalid model index")
        exit(-2)

    event_segmentation(MODEL_IDS[model_idx])
