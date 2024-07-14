from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.keys import HUGGING_FACE_TOKEN

if torch.backends.mps.is_built():
    print('mps is used.')
    mps_device = torch.device("mps")

class SequentialityModel:
    def __init__(self, model_name):
        self.sentence = ""  # this is what sequentiality is calculated on
        self.stem = ""  # this is a slice of the sentence - used for context for model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, use_safetensors = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          token=HUGGING_FACE_TOKEN,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map=mps_device,
                                                          use_safetensors=True).to("mps")
        # turn off gradient descent
        torch.set_grad_enabled(False)

    def set_stem(self, string : str):
        self.stem = string

    def k_likelihood(self, k : int, verbose : bool = False):
        """Function that returns the likelihoods of the top k words in the current stem"""
        if self.stem == "":
            return None

        # tokenize to ids
        input_ids = self.tokenizer.encode(self.stem, return_tensors="pt").to("mps")

        # call model() to get logits
        logits = self.model(input_ids).logits

        # only care about the last projection in the last batch
        logits = logits[-1, -1]

        # softmax() to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # keep only the top 20
        probs, ids = torch.topk(probs, k)

        # convert ids to tokens
        texts = self.tokenizer.convert_ids_to_tokens(ids)

        ret = zip(probs, texts)

        if verbose:  # debug statement
            for prob, text in ret:
                print(f"{prob:.4f}: \"{text}\"")

        return ret


if __name__ == "__main__":
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct")
    model.set_stem("Hi! Nice to meet")
    model.k_likelihood(20, True)
