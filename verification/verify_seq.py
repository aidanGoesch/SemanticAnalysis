import sys
import numpy as np
import pandas as pd
from src.sequentiality import SequentialityModel


def load_data():
    df = pd.read_csv("./data/truncated_data.csv")
    return df


def write_data(file_name, data : pd.DataFrame):
    data.to_csv(f"./data/calculated_values/{file_name}")


def verify_data(partition_id:int, participant_id:int, recall_length:int):
    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                            "scalar_text_sequentiality",
                                            "sentence_total_sequentialities",
                                            "sentence_contextual_sequentialities",
                                            "sentence_topic_sequentialities",
                                            "story",
                                            "recAgnPairId",
                                            "recImgPairId"])
    data = load_data()

    vec = data.iloc[partition_id + participant_id]

    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct",
                               topic="a conversation with a doctor",
                               recall_length=recall_length)

    seq = model.calculate_text_sequentiality(vec.story)
    sequentialities.loc[0] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

    write_data(f"{recall_length}/{partition_id + participant_id}.csv", sequentialities)


def test_data(partition_id:int, participant_id:int):
    recall_length = 3 # hard code this to reduce unnecessary computations

    sequentialities = pd.DataFrame(columns=["AssignmentId",
                                            "scalar_text_sequentiality",
                                            "sentence_total_sequentialities",
                                            "sentence_contextual_sequentialities",
                                            "sentence_topic_sequentialities",
                                            "story",
                                            "recAgnPairId",
                                            "recImgPairId"])
    data = load_data()

    vec = data.iloc[partition_id + participant_id]

    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct",
                               topic="a conversation with a doctor",
                               recall_length=recall_length)

    seq = model.calculate_text_sequentiality(vec.story)
    sequentialities.loc[0] = [vec.AssignmentId] + seq + [vec.story, vec.recAgnPairId, vec.recImgPairId]

    write_data(f"testing/{partition_id + participant_id}.csv", sequentialities)

if __name__ == "__main__":
    # verify_data(*sys.argv)
    load_data()
