from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from src.keys import HUGGING_FACE_TOKEN
import re
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

if torch.backends.mps.is_built():  # Apple Silicon
    print('mps is used.')
    mps_device = torch.device("mps")
elif torch.backends.cuda.is_built():  # Real GPU
    print('cuda is used.')
    mps_device = torch.device("cuda")
else:  # If all else fails
    print('cpu is used.')
    mps_device = torch.device("cpu")


# torch.set_default_dtype(torch.bfloat16)
torch.set_float32_matmul_precision('high')

class SequentialityModel:
    def __init__(self, model_name : str, topic : str, recall_length=4) -> None:
        self.sentences = []

        self.recall_length = recall_length

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

        # Pad all text with _
        self.context_string = f"_condition every word on this topic: <TOPIC>{topic}<END_TOPIC> "

    def _to_tokens_and_logprobs(self, text: str) -> list[list[tuple[int, float]]]:
        input_text = self.context_string + text
        input_ids = self.tokenizer(input_text, padding=True, return_tensors="pt").input_ids.to(mps_device)
        
        with torch.inference_mode(): #optimize for inference
            outputs = self.model(input_ids)

        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            token_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token.item() not in self.tokenizer.all_special_ids:
                    token_sequence.append((token.item(), p.item()))
            batch.append(token_sequence)
        return batch

    def print_token_ids_and_strings(self, token_ids: list[int]):
        print("Query token sequence:")
        for token_id in token_ids:
            token_str = self.tokenizer.decode([token_id])
            print(f"Token: {token_str!r:10} | ID: {token_id:6}")

    @staticmethod
    def _find_subsequence(query: list[int], full_sequence: list[tuple[int, float]]) -> int:
        """Return the starting index in full_sequence where query is found, or -1 if not found."""
        n = len(query)
        for i in range(len(full_sequence) - n + 1):
            if all(full_sequence[i + j][0] == query[j] for j in range(n)):
                return i
        return -1

    def _process_tokens_and_logprobs(self, query_token_ids: list[int], tokens_and_logprobs: list[tuple[int, float]]) -> float:
        start_idx = SequentialityModel._find_subsequence(query_token_ids, tokens_and_logprobs)
        if start_idx == -1:
            # Debug: print the sequences to see why matching failed
            print("Query token IDs:", query_token_ids)
            self.print_token_ids_and_strings(query_token_ids)
            print("Full sequence token IDs:", [t for t, _ in tokens_and_logprobs])
            self.print_token_ids_and_strings([t for t, _ in tokens_and_logprobs])
            return 0
        # Sum only over the tokens corresponding to the query (not the rest of the sequence)
        return sum(p for _, p in tokens_and_logprobs[start_idx:start_idx+len(query_token_ids)])

    def _calculate_contextual_sequentiality(self, sentence : str, sentence_tokens : list[str], i : int,  h : int, verbose : bool = False) -> float:
        """
        Calculate the contextually dependent sequentiality of a sentence.

        :param sentence: raw input sentence
        :param sentence_tokens: tokenized version of sentence
        :param i: index of current sentence
        :param h: number of sentences to use for context

        :return: contextual sequentiality value - Log(P(sentence | previous h sentences ^ topic))
        :rtype: float
        """
        raw_sequentiality = 0
        if i - h < 0:
            context = " ".join(self.sentences[:i])

        else:
            context = " ".join(self.sentences[i - h:i])

        if len(context) == 0:  # beginning of the text - prevents random period at the front of the text
            input_text = sentence
        else:
            input_text = context + " " + sentence

        tokens_and_logprobs = self._to_tokens_and_logprobs(input_text)[0]
        
        return self._process_tokens_and_logprobs(sentence_tokens, tokens_and_logprobs)

    def _calculate_topic_sequentiality(self, sentence : str, sentence_tokens : list[str], verbose : bool = False) -> float:
        """
        Calculate the sequentiality of a sentence given only a topic

        :param sentence: raw input sentence,
        :param sentence_tokens: tokenized version of sentence

        :return: topic sequentiality value - Log(P(sentence | topic))
        :rtype: float
        """
        # Tokenize the full text (which is context + sentence)
        full_text = self.context_string + sentence
        tokens_and_logprobs = self._to_tokens_and_logprobs(full_text)[0]
        return self._process_tokens_and_logprobs(sentence_tokens, tokens_and_logprobs)

    def _calculate_sentence_sequentiality(self, sentence : str, i: int, verbose : bool = False) -> list[float]:
        """

        Calculates the sequentiality of a given sentence by subtracting the context dependent sequentiality from
        the purely topic driven version.

        :param sentence: raw input sentence
        :param i: index of current sentence
        :param verbose: debug

        :return: [total_sentence_sequentiality, contextual_sequentiality, topic_sequentiality]
        :rtype: list[float]
        """
        # Tokenize the context string separately.
        context_ids = self.tokenizer.encode(self.context_string, add_special_tokens=False)
        
        # Tokenize the full input (context + sentence)
        full_text = self.context_string + sentence
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # Extract the sentence tokens by slicing off the context portion.
        sentence_token_ids = full_ids[len(context_ids):]

        # log probs
        topic_sequentiality = self._calculate_topic_sequentiality(sentence, sentence_token_ids)
        contextual_sequentiality = self._calculate_contextual_sequentiality(
            sentence=sentence,
            sentence_tokens=sentence_token_ids,
            i=i,
            h=self.recall_length,
            verbose=verbose
        )

        if verbose:
            print(f"topic sequentiality: {topic_sequentiality}")
            print(f"context sequentiality: {contextual_sequentiality}")
            print("Sentence token IDs:")
            print(sentence_token_ids)
            # Optionally print decoded tokens:
            print("Sentence token sequence:")
            for token_id in sentence_token_ids:
                print(f"Token: {self.tokenizer.decode([token_id])!r} | ID: {token_id}")
            print(f"Sentence: {sentence}")
        
        # Normalize by the number of tokens, if desired.
        return [(topic_sequentiality - contextual_sequentiality) / -len(sentence_token_ids), contextual_sequentiality, topic_sequentiality]

    def calculate_text_sequentiality(self, text : str, verbose : bool = False) -> list[float | list]:
        """
        Function that calculates the total sequentiality of a text

        :param text: entire input text
        :param verbose: debug

        :return: [total_text_sequentiality, total_sentence-level_sequentiality, contextual_sentence-level_sequentiality, topic_sentence-level_sequentiality]
        :rtype: list[float | list]
        """
        split_text = re.split(r'(?<!\.\.\.)[\.\?\!](?!\.)\s*', text)

        self.sentences = []
        for i in range(0, len(split_text) - 1, 2):
            sentence = split_text[i].strip() + split_text[i + 1]
            self.sentences.append(sentence)

        total_sequentialities = []
        contextual_sequentialities = []
        topic_sequentialities = []

        for i, sentence in enumerate(self.sentences):
            if sentence == "": continue

            total, contextual, topic = self._calculate_sentence_sequentiality(sentence, i)
            total_sequentialities.append(total)
            contextual_sequentialities.append(contextual)
            topic_sequentialities.append(topic)

        return [np.mean(total_sequentialities), total_sequentialities, contextual_sequentialities, topic_sequentialities]


if __name__ == "__main__":
    # model = SequentialityModel("microsoft/Phi-3-mini-4k-instruct", topic="a conversation with a doctor")
    model = SequentialityModel("SakanaAI/TinySwallow-1.5B-Instruct", topic="a conversation with a doctor")
    # model = SequentialityModel("meta-llama/Llama-3.3-70B-Instruct", topic="a conversation with a doctor")
    print(f"\nshould be lower  : {model.calculate_text_sequentiality("There are two bison standing next to each other. They seem to be friends. Why is this not working.", False)}")
    print(f"\nshould be higher : {model.calculate_text_sequentiality("I broke my arm. It hurts a lot, and I don't know if it'll ever heal. When I looked down, I could see the bone sticking out.", False)}")

