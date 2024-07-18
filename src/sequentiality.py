from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import string
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

        # tokenize to ids
        input_ids = self.tokenizer.encode(self.stem, return_tensors="pt").to(mps_device)

        # call model() to get logits
        logits = self.model(input_ids).logits

        # only care about the last projection in the last batch
        logits = logits[-1, -1]

        # softmax() to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float16)

        # keep only the top 20
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
                if key.strip("▁").lower() == query_key.lower():
                    return likelihood_dict[key]

        return 0  # Next word is not in the likelihood dict


    def process_sentence(self, sentence: str) -> str:
        """Function that gets rid of punctuation and split it to return a list of tokens"""

        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        ids = self.tokenizer.encode(sentence, return_tensors="pt").to(mps_device)

        # This causes a bug
        translated_tokens = self.tokenizer.decode(ids)

        print(translated_tokens)


    def calculate_sequentiality(self, sentence : str, verbose : bool = False) -> int:
        """Returns the sum of the likelihoods of each word in a sentence"""
        fragments = SequentialityModel.process_sentence(sentence)

        total_likelihood = 0

        for i in range(len(fragments)):  # makes it so that you start with a seed word and then finish with the last word
            if i == 0: continue

            self.stem = " ".join(fragments[:i])

            if verbose:
                print(f"DEBUG: iteration {i} / {len(fragments)} started - stem: {self.stem}")

            likelihood_dict = self.k_likelihood(100, False)

            # TODO: make this work with token strings instead of words
            likelihood = self.likelihood_from_dict(likelihood_dict, fragments[i])

            if verbose:
                print(f"\nlikelihood of '{fragments[i]}' given stem '{self.stem}' = {likelihood}\n")

            if verbose:
                print(f"DEBUG: iteration {i} / {len(fragments)} ended")
                print(f"DEBUG: likelihood of '{fragments[i]}': {likelihood}")

            total_likelihood += likelihood

        return total_likelihood




if __name__ == "__main__":
    model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct")
    model.process_sentence("The apple. fell, from the tree!")
    # print(f"total likelihood = {model.calculate_sequentiality("It is nice to meet you!", True)}")
    # print(f"total likelihood = {model.calculate_sequentiality("It is nice to meet gorilla!", True)}")
    # model.set_stem("Hi! Nice to meet")
    # # model.k_likelihood(50, True)
    #
    # model.calculate_sequentiality("Hi! Nice to meet you", True)


    # test_dict = {"▁meet": 0.83251953, "▁see": 0.1126709, "▁hear": 0.01187897, "▁have": 0.00924683, "▁finally": 0.00720215, "▁chat": 0.0056076, "▁talk": 0.00160694, "▁virt": 0.00125122, "▁make": 0.00097466, "▁'": 0.00075912, "▁interact": 0.00059128, "▁Me": 0.00027919, "▁virtual": 0.00021756, "▁catch": 0.0001694, "▁go": 0.00013196, "▁learn": 0.00010276, "▁Connect": 7.999e-05, "▁": 6.229e-05, "▁meeting": 4.852e-05, "▁reach": 3.779e-05, "▁beat": 2.944e-05, "▁Virtual": 2.295e-05, ",": 1.788e-05, "▁conquer": 1.389e-05, "▁beh": 1.085e-05, }
    # print(model.likelihood_from_dict(test_dict, "hear"))
    # EO6
    # import random
    #
    # count = 0
    # for i in range(100000):
    #     if sum(random.randint(1, 6) for j in range(5)) > 10:
    #         count += 1
    #
    # prob = count / 100000
    # print(prob)

