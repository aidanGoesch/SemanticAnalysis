from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from src.keys import HUGGING_FACE_TOKEN
import re
import json
import time
# from torch._dynamo import optimize
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
                                                       padding_side="left",
                                                       use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          token=HUGGING_FACE_TOKEN,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map=mps_device,
                                                          use_safetensors=True).to(mps_device)
        
        self.model.generation_config.cache_implementation = "static"

        # try:
        #     self.model = torch.compile(self.model, backend="inductor", 
        #                 mode="reduce-overhead",  # Much more conservative
        #                 fullgraph=False,         # Allow partial compilation
        #                 dynamic=True)           # Handle dynamic shapes better
        #     print("Model Compiled")
        # except Exception as e:
        #     print(f"Model not compiled: {e}")
        
        self.model.eval()  # Ensure model is in evaluation mode
        torch.set_grad_enabled(False)  # Disable gradient calculation

        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Pad all text with _
        self.context_string = f"_condition every word on this topic: <TOPIC>{topic}<END_TOPIC> "

    def _to_tokens_and_logprobs(self, text: str) -> list[list[tuple[int, float]]]:
        start_time = time.perf_counter() 
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
        special_ids = set(self.tokenizer.all_special_ids)
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            token_sequence = []
            for token, p in zip(input_sentence, input_probs): # get a list of tokens that and logprobs for a given sentence
                if token.item() not in special_ids:
                    token_sequence.append((token.item(), p.item()))
            batch.append(token_sequence)
        
        # print(f"model infernce time: {time.perf_counter() - start_time}")
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
        start_time = time.perf_counter()
        start_idx = SequentialityModel._find_subsequence(query_token_ids, tokens_and_logprobs)
        if start_idx == -1:
            # Debug: print the sequences to see why matching failed
            print("Query token IDs:", query_token_ids)
            self.print_token_ids_and_strings(query_token_ids)
            print("Full sequence token IDs:", [t for t, _ in tokens_and_logprobs])
            self.print_token_ids_and_strings([t for t, _ in tokens_and_logprobs])
            return 0
        # print(f"processing time: {time.perf_counter() - start_time}")
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
        if len(sentence) == 0:  # artifact of new regex - shouldn't change anything
            return 0

        # Tokenize the context string separately.
        context_ids = self.tokenizer.encode(self.context_string, add_special_tokens=False)

        if hasattr(self, 'token_cache') and sentence in self.token_cache:
            sentence_token_ids = self.token_cache[sentence]
        else:
            # Existing tokenization logic
            context_ids = self.tokenizer.encode(self.context_string, add_special_tokens=False)
            full_text = self.context_string + sentence
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            sentence_token_ids = full_ids[len(context_ids):]
            
            # Cache for future use
            if not hasattr(self, 'token_cache'):
                self.token_cache = {}
            self.token_cache[sentence] = sentence_token_ids
        
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
    
    def load_tokens_to_cache(self, tokenized_data_path):
        """
        Load pre-tokenized sentences into the token cache
        
        :param tokenized_data_path: Path to CSV with tokenized data
        """
        # Initialize token cache if it doesn't exist
        if not hasattr(self, 'token_cache'):
            self.token_cache = {}
        
        # Load the tokenized data
        tokenized_df = pd.read_csv(tokenized_data_path)
        
        print(f"Loading {len(tokenized_df)} stories into token cache...")
        loaded_tokens = 0
        
        # Process each story
        for i in range(len(tokenized_df)):
            story = tokenized_df.iloc[i].story
            tokenized_sentences = json.loads(tokenized_df.iloc[i].tokenized_sentences)
            
            # Split text to get the same sentences as in original tokenization
            split_text = re.split(r'(?<!\.\.\.)[\.\?\!](?!\.)\s*', story)
            processed_sentences = []
            
            for j in range(0, len(split_text) - 1, 2):
                if j+1 < len(split_text):
                    sentence = split_text[j].strip() + split_text[j + 1]
                    processed_sentences.append(sentence)
            
            # Add each sentence and its tokens to the cache
            for sentence, tokens in zip(processed_sentences, tokenized_sentences):
                if sentence and tokens:  # Skip empty entries
                    self.token_cache[sentence] = tokens
                    loaded_tokens += 1
        
        print(f"Loaded {loaded_tokens} tokenized sentences into cache")
    

    def _tokenize_with_cache(self, sentence):
        """Tokenize with caching for repeated sentences."""
        if not hasattr(self, 'token_cache'):
            self.token_cache = {}
            
        if sentence in self.token_cache:
            return self.token_cache[sentence]
        
        # Context string tokenization (happens once)
        if not hasattr(self, '_context_token_ids'):
            self._context_token_ids = self.tokenizer.encode(self.context_string, add_special_tokens=False)
        
        # Tokenize full text
        full_text = self.context_string + sentence
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # Extract just the sentence tokens
        sentence_token_ids = full_ids[len(self._context_token_ids):]
        
        # Cache the result
        self.token_cache[sentence] = sentence_token_ids
        return sentence_token_ids

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

