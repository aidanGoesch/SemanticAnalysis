from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from collections import Counter
from src.keys import HUGGING_FACE_TOKEN

if torch.backends.mps.is_built():  # Apple Silicon
    print('mps is used.')
    mps_device = torch.device("mps")
elif torch.backends.cuda.is_built():  # Real GPU
    print('cuda is used.')
    mps_device = torch.device("cuda")
else:  # If all else fails
    print('cpu is used.')
    mps_device = torch.device("cpu")


class SequentialityModel:
    def __init__(self, model_name) -> None:
        self.sentence = ""  # this is what sequentiality is calculated on
        self.stem = ""  # this is a slice of the sentence - used for context for model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, use_safetensors = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          token=HUGGING_FACE_TOKEN,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map=mps_device,
                                                          use_safetensors=True).to(mps_device)
        # turn off gradient descent
        torch.set_grad_enabled(False)

    def set_stem(self, string : str) -> None:
        """Set method for self.stem"""
        self.stem = string

    def k_likelihood(self, k : int, verbose : bool = False) -> dict[str, int]:
        """Function that returns the likelihoods of the top k words in the current stem"""
        if self.stem == "":
            return None

        # turn text tokens into ids and convert to tensor
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.stem)]).to(mps_device)

        # call model() to get logits
        logits = self.model(input_ids).logits

        # only care about the last projection in the last batch
        logits = logits[-1, -1]

        # softmax() to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float16)

        # keep only the top k
        probs, ids = torch.topk(probs, k)

        # convert ids to tokens
        texts = self.tokenizer.convert_ids_to_tokens(ids)

        probability_dict = dict(zip(texts, probs))

        if verbose:  # debug statement
            for text, prob in probability_dict.items():
                print(f"{prob:.8f}: \"{text}\"")

        return probability_dict

    @staticmethod
    def likelihood_from_dict(likelihood_dict: dict, query_key : str) -> int:
        """Returns the likelihood of a word if it is a substring in likelihood_dict or 0 otherwise"""
        for key in likelihood_dict.keys():
            if query_key.lower() in key.lower():
                if key.lower() == query_key.lower():
                    return likelihood_dict[key]

        return 0  # Next word is not in the likelihood dict


    def process_sentence(self, sentence: str, verbose : bool = False) -> (list[str], list[str]):
        """Function that gets rid of punctuation and split it to return a list of tokens"""
        if sentence == "":
            return ([], [])

        # sentence = sentence.translate(str.maketrans('', '', string.punctuation)) + "."

        ids = self.tokenizer.encode(sentence, return_tensors="pt").to(mps_device)

        translated_tokens = self.tokenizer.convert_ids_to_tokens(ids[0])

        if verbose:
            print(translated_tokens)

        return sentence.split(), translated_tokens

    def calculate_single_sentence_sequentiality(self, sentence: str, verbose : bool = False) -> float:
        """Function that returns the negative log likelihood of a sentence"""
        fragments, tokens = self.process_sentence(sentence)

        total_likelihood = 0
        epsilon = 1e-10  # used for epslion smoothig - prevents log(0)

        for i in range(len(tokens)):  # makes it so that you start with a seed word and then finish with the last word
            if i == 0: continue

            self.stem = tokens[:i]

            if verbose:
                print(f"\nDEBUG: iteration {i} / {len(tokens)} started - stem: {self.stem}")

            likelihood_dict = self.k_likelihood(100, False)

            likelihood = self.likelihood_from_dict(likelihood_dict, tokens[i])

            if verbose:
                print(f"\nlikelihood of '{tokens[i]}' given stem '{self.stem}' = {likelihood}\n")

            if verbose:
                print(f"DEBUG: iteration {i} / {len(tokens)} ended")
                print(f"DEBUG: likelihood of '{tokens[i]}': {likelihood}\n")

            if not isinstance(likelihood, int):
                likelihood = likelihood.cpu().numpy()

            if likelihood == 0:
                likelihood = epsilon

            total_likelihood += np.log10(likelihood)

        return -total_likelihood / len(tokens)  # normalize output for length of sentence

    def calculate_sequentiality(self, text : str, verbose : bool = False) -> float:
        """Returns the sum of the likelihoods of each word in a sentence. This may need to change
        depending on how the transcription works and spereates sentences."""
        # TODO: condition the likelihood on a topic
        contextual_nll, topic_nll = [], []
        if Counter(text)["."] > 1:  # Case where there are multiple sentences in the text
            text = text.split(".")

            for i in range(len(text)):
                if text[i] == "": continue

                contextual_nll.append(self.calculate_single_sentence_sequentiality(". ".join(text[:i + 1]) + ".", verbose))
                topic_nll.append(self.calculate_single_sentence_sequentiality(text[i], verbose))

                if verbose:
                    print(f"\nDEBUG: contextual nll of '{text[i]}': {contextual_nll[i]}")
                    print(f"DEBUG: topic nll of '{text[i]}': {topic_nll[i]}\n")

            return contextual_nll[-1] + topic_nll[-1]  # summation because they are both already NLLs

        else:
            return self.calculate_single_sentence_sequentiality(text, verbose)



if __name__ == "__main__":
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct")
    print(f"\ntotal NLL of 'I really like you. I never want to see you again.'= {model.calculate_sequentiality("I really like you. I never want to see you again.", False)}")
    # print(f"\ntotal likelihood = {model.calculate_sequentiality("It is nice to meet gorilla.", True)}")

    print(f"\ntotal NLL of 'I broke my wrist. It hurt a lot.' {model.calculate_sequentiality("I broke my wrist. It hurt a lot.", False)}")
