import numpy as np
import regex as re
import tensorflow_hub as hub

from numpy.linalg import norm



class SequentialityEmbeddingModel():
    def __init__(self, embedding_function, topic:str, recall_length:int):
        self.embedding_function = embedding_function
        self.recall_length = recall_length

        self.topic = topic
        self.default_topic = topic
        self.topic_string = self.topic_string = f"_condition every word on this topic: <TOPIC>{self.topic}<END_TOPIC> "
    
    def set_topic(self, topic: str):
        """Method that sets the topic of the model"""
        self.topic = topic
        self.topic_string = f"_condition every word on this topic: <TOPIC>{topic}<END_TOPIC> "
    
    def _cosine_similarity(self, vec1:list, vec2:list) -> float:
        """Function that calculates the cosine similarity between two vectors"""
        vec1_flat = vec1.numpy().flatten()
        vec2_flat = vec2.numpy().flatten()
        
        return np.dot(vec1_flat, vec2_flat) / (norm(vec1_flat) * norm(vec2_flat))

    def _calculate_topic_sequentiality(self, sentence:str, sentence_embedding:list[int], verbose:bool = False) -> float:
        """
        Calculates the cosine similarity between a sentence embedding and the embedding of the topic vector.

        :param sentence: raw input sentence,
        :param sentence_tokens: tokenized version of sentence

        :return: topic sequentiality value - Log(P(sentence | topic))
        :rtype: float
        """
        
        topic_embedding = self.embedding_function([self.topic_string])

        similarity = self._cosine_similarity(topic_embedding, sentence_embedding)

        if verbose:
            print(f"topic embedding similiarity of sentence: {sentence}\n={similarity}")

        return similarity

    def _calculate_contextual_sequentiality(self, sentence:str, sentence_embedding:list[str], i:int,  h:int, verbose:bool = False) -> float:
        """
        Calculate the cosine similarity between a sentences embedding and the embedding of the topic and context

        :param sentence: raw input sentence
        :param sentence_tokens: tokenized version of sentence
        :param i: index of current sentence
        :param h: number of sentences to use for context

        :return: contextual sequentiality value - Log(P(sentence | previous h sentences ^ topic))
        :rtype: float
        """
        # get the correct context string according to h
        if i - h < 0:
            context = " ".join(self.sentences[:i])
        else:
            context = " ".join(self.sentences[i - h:i])
        
        full_text = self.topic_string + context

        full_text_embedding = self.embedding_function([full_text])

        similarity = self._cosine_similarity(full_text_embedding, sentence_embedding)

        if verbose:
            print(f"contextual embedding similiarity of sentence: {sentence}\n={similarity}")

        return similarity

    def _calculate_sentence_sequentiality(self, sentence:str, i:int, verbose:bool=False):
        """
        Calculates the sequentiality of a given sentence by subtracting the context dependent sequentiality from
        the purely topic driven version.

        :param sentence: raw input sentence
        :param i: index of current sentence
        :param verbose: debug

        :return: [total_sentence_sequentiality, contextual_sequentiality, topic_sequentiality]
        :rtype: list[float]
        """

        sentence_embedding = self.embedding_function([sentence])



        topic_sequentiality = self._calculate_topic_sequentiality(sentence, sentence_embedding)
        contextual_sequentiality = self._calculate_contextual_sequentiality(
            sentence=sentence,
            sentence_embedding=sentence_embedding,
            i=i,
            h=self.recall_length,
            verbose=verbose
        )

        return np.mean([topic_sequentiality, contextual_sequentiality]), contextual_sequentiality, topic_sequentiality

         

    def calculate_text_sequentiality(self, text:str, topic:str=None, verbose:bool=False) -> list[float | list]:  # TODO: Change this so that it uses the embeddings
        """
        Function that calculates the total sequentiality of a text

        :param text: entire input text
        :param topic: a topic to condition the text on
        :param verbose: debug

        :return: [total_text_sequentiality, total_sentence-level_sequentiality, contextual_sentence-level_sequentiality, topic_sentence-level_sequentiality]
        :rtype: list[float | list]
        """

        if topic is not None:
            self.set_topic(topic)
        else:
            self.set_topic(self.default_topic)

        # split text into sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)

        self.sentences = [s.strip() for s in sentences if s.strip()]

        total_sequentialities = []
        contextual_sequentialities = []
        topic_sequentialities = []

        for i, sentence in enumerate(self.sentences):
            if sentence == "": continue

            total, contextual, topic = self._calculate_sentence_sequentiality(sentence, i)
            total_sequentialities.append(total)
            contextual_sequentialities.append(contextual)
            topic_sequentialities.append(topic)

        return [
        np.mean(total_sequentialities),
        total_sequentialities,
        contextual_sequentialities,
        topic_sequentialities
    ]

    



if __name__ == "__main__":
    print("loading model...")
    embed = hub.load(
    "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
    )
    print("model loaded")

    model = SequentialityEmbeddingModel(embed, "a conversation with a doctor", 4)
    print("format: topic_sequentiality, contextual_sequentiality")
    print(f"\nshould be lower  : {model.calculate_text_sequentiality("There are two bison standing next to each other. They seem to be friends. Why is this not working.", False)[:2]}")
    print(f"\nshould be higher : {model.calculate_text_sequentiality("I broke my arm. It hurts a lot, and I don't know if it'll ever heal. When I looked down, I could see the bone sticking out.", False)[:2]}")



