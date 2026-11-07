"""
Calculate FilmFest sequentiality for OLMo and Llama on HPC.

Same setup as datasets/filmfest/adjusted_filmfest_dataset_components.csv:
10 stories x 2 topic conditions (naive title / clairvoyant summary), history lengths 2, 4, 6, 8.

Usage: python3 run_filmfest.py <model_index>
"""
import os
import sys

import pandas as pd

from src.sequentiality import calculate_sequentiality

MODEL_IDS = ["allenai/OLMo-2-1124-13B",
             "meta-llama/Llama-3.1-8B-Instruct"]

HISTORY_LENGTHS = [2, 4, 6, 8]

STORY_DIR = "./datasets/filmfest/stories"
SAVE_DIR = "./outputs/ensemble/filmfest"

# story number -> (naive topic, clairvoyant topic)
TOPICS = {
    1:  ("A short story titled 'Catch me if you Can'", "A short story about a famous con man being captured"),
    2:  ("A short story titled 'The Record'",          "A short story about a woman rapidly aging through her entire life"),
    3:  ("A short story titled 'The Boyfriend'",       "A short story about a robot woman replacing her robot boyfriend"),
    4:  ("A short story titled 'The Shoe'",            "A short story about a man and a woman mimicking each other until he is hit by a car"),
    5:  ("A short story titled 'Keith Reynolds'",      "A short story about a man hoping for a promotion who ends up killing a coworker"),
    6:  ("A short story titled 'The Rock'",            "A short story about a mountain becoming space debris"),
    7:  ("A short story titled 'The Prisoner'",        "A short story about a man trapped in a strange village"),
    8:  ("A short story titled 'The Black Hole'",      "A short story about a man getting trapped in a safe"),
    9:  ("A short story titled 'Post-it Love'",        "A short story about two coworkers creating a mural of themselves"),
    10: ("A short story titled 'Bus Stop'",            "A short story about a man at a bus stop that turns into a dog"),
}


def load_story(story_number:int) -> str:
    """Stories are stored one sentence per line; join them into a single text."""
    with open(f"{STORY_DIR}/story{story_number}.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return " ".join(lines)


def filmfest(model_id:str):
    texts, topics, stories, story_types = [], [], [], []
    for story_number, (naive_topic, clairvoyant_topic) in TOPICS.items():
        story = load_story(story_number)
        for story_type, topic in [("naive", naive_topic), ("clairvoyant", clairvoyant_topic)]:
            texts.append(story)
            topics.append(topic)
            stories.append(story_number)
            story_types.append(story_type)

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=HISTORY_LENGTHS,
                                     text_input=texts,
                                     topics=topics)

    if output.empty:
        print(f"No output for {model_id}, nothing saved")
        exit(-1)

    # rows are ordered history length -> text, so story labels repeat once per history length
    output["story"] = stories * len(HISTORY_LENGTHS)
    output["story_type"] = story_types * len(HISTORY_LENGTHS)

    os.makedirs(SAVE_DIR, exist_ok=True)
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"{SAVE_DIR}/{safe_model_name}.csv", index=False)

    # long format: one row per sentence, same columns as adjusted_filmfest_dataset_components.csv
    rows = []
    for _, row in output.iterrows():
        for i, (total, contextual, topic_seq) in enumerate(zip(row["sentence_total_sequentialities"],
                                                               row["sentence_contextual_sequentialities"],
                                                               row["sentence_topic_sequentialities"])):
            rows.append([row["topic"], model_id, row["history_length"], row["story"], i,
                         total, contextual, topic_seq, row["story_type"]])

    long_output = pd.DataFrame(rows, columns=["topic", "model", "history", "story", "sentence",
                                              "total_sequentiality", "contextual_sequentiality",
                                              "topic_sequentiality", "story_type"])
    long_output.to_csv(f"{SAVE_DIR}/{safe_model_name}_components.csv", index=False)

    print(f"Saved {len(output)} stories / {len(long_output)} sentences to {SAVE_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_filmfest.py <model_index>")
        for i, model in enumerate(MODEL_IDS):
            print(f"  {i}: {model}")
        exit(-2)

    model_idx = int(sys.argv[1])
    if model_idx not in range(len(MODEL_IDS)):
        print("invalid model index")
        exit(-2)

    filmfest(MODEL_IDS[model_idx])
