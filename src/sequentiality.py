from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from collections import Counter
from src.keys import HUGGING_FACE_TOKEN
import re

if torch.backends.mps.is_built():  # Apple Silicon
    print('mps is used.')
    mps_device = torch.device("mps")
elif torch.backends.cuda.is_built():  # Real GPU
    print('cuda is used.')
    mps_device = torch.device("cuda")
else:  # If all else fails
    print('cpu is used.')
    mps_device = torch.device("cpu")

# TODO: change to vector embedding?
# get rid of stop words?

# number of previous sentences used to calculate context dependent sequentiality
CALL_BACK = 4


class SequentialityModel:
    def __init__(self, model_name : str, topic : str) -> None:
        self.sentence = ""  # this is what sequentiality is calculated on
        self.stem = ""  # this is a slice of the sentence - used for context for model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       token=HUGGING_FACE_TOKEN,
                                                       use_safetensors=True,
                                                       padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          token=HUGGING_FACE_TOKEN,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map=mps_device,
                                                          use_safetensors=True).to(mps_device)

        self.model.config.pad_token_id = self.model.config.eos_token_id
        # turn off gradient descent
        torch.set_grad_enabled(False)

        self.context_string = f"The text after the colon is dependent on the topic of {topic}: "
        self.context_tokens = self.tokenizer.encode(self.context_string)

        self.memoize = {}

    def _to_tokens_and_logprobs(self, input_texts):
        # manually pad every input string with an underscore
        input_ids = self.tokenizer("_" + input_texts, padding=True, return_tensors="pt").input_ids.to(mps_device)
        outputs = self.model(input_ids)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in self.tokenizer.all_special_ids:
                    text_sequence.append((self.tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        return batch



    def calculate_contextual_sequentiality(self, sentence : str, i : int,  h : int, verbose : bool = False) -> float:
        """Calculate the contextually dependent sequentiality of a sentence."""
        raw_sequentiality = 0
        if i - h < 0:
            context = ". ".join(self.sentences[:i])

        else:
            context = ". ".join(self.sentences[i - h:i])

        if len(context) == 0:  # beginning of the text - prevents random period at the front of the text
            input_text = sentence + "."
        else:
            input_text = context + ". " + sentence + "."

        tokens_and_logprobs = self._to_tokens_and_logprobs(input_text)[0]

        tokenized_sentence = self.tokenizer.tokenize(sentence)

        first_token = tokenized_sentence[0].strip("▁")

        for j, tmp in enumerate(reversed(tokens_and_logprobs)):
            if tmp[0].lower() == first_token.lower():
                raw_sequentiality = sum(map(lambda x: x[1], tokens_and_logprobs[j:]))
                break

        return raw_sequentiality

    def calculate_topic_sequentiality(self, sentence : str, verbose : bool = False) -> float:
        """Calculate the sequentiality of a sentence given only a topic"""
        pass

    def calculate_sequentiality(self, sentence : str, i: int, verbose : bool = False) -> float:
        """Calculates the sequentiality of a given sentence by subtracting the context dependent sequentiality from
        the purely topic driven version."""
        tokenized_sentence = self.tokenizer.tokenize(sentence)

        topic_sequentiality = self.calculate_topic_sequentiality(sentence)
        contextual_sequentiality = self.calculate_contextual_sequentiality(sentence, i, CALL_BACK, False)

        return -(topic_sequentiality - contextual_sequentiality) / len(tokenized_sentence)

    def calculate_total_sequentialty(self, text : str, verbose : bool = False) -> float:
        self.sentences = re.split('[\.\?\!]\s*', text)
        sequentialities = []

        for i, sentence in enumerate(self.sentences):
            if sentence == "": break

            sequentialities.append(self.calculate_sequentiality(sentence, i, CALL_BACK, False))

        return np.mean(sequentialities)


if __name__ == "__main__":
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a conversation with a nurse")
    # print(f"\ntotal NLL of 'There are two bison standing next to each other. They seem to be friends.'= {model.calculate_sequentiality("There are two bison standing next to each other. They seem to be friends.", False)}")
    # print(f"\ntotal likelihood = {model.calculate_sequentiality("It is nice to meet you.", False)}")
    import time

    # model.sentences = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    #
    # model.calculate_sequentiality("test", 1, CALL_BACK, False)

    # start = time.time()
    print(f"\ntotal NLL of test scene= {model.calculate_total_sequentialty('"THE BOYFRIEND" in bold white text fades in on a black screen before fading out. The letters of "high maintenance" appear in the center of the screen one by one in white text. A simple jingle plays in the background.', False)}")
    # print("time to run: ", time.time() - start)
