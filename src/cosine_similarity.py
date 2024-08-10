
import tensorflow_hub as hub
import numpy as np
from numpy.linalg import norm


def compute_cosine_similarity(string1 : str, string2 : str, verbose : bool = False) -> float:
    embed = hub.load(
        "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
    embeddings = embed([string1, string2])

    if verbose:
        print(embeddings)

    # compute the cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))

    return similarity


if __name__ == "__main__":
    description1 = "I remember the first time that I had alcohol very well. It was me and my friends at a bar and we laughed and had a jolly time!"
    description2 = "The most memorable event I had involving alcohol has to be that time when I was in a bar in mexico. I was alone and drank until I couldn't find my feet"

    print(compute_cosine_similarity(description1, description2, True))
