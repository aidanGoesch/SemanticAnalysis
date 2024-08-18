import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class ExploratoryAnalysis:
    def __init__(self, text : str):
        self.text = text

        # dict that is updated in every analysis member function
        self.analysis_results = {}


    def create_report(self):
        """Function that calls performs analysis methods and compiles 2 dictionaries corresponding to the initial
        description and memorable description"""
        pass

    def bag_of_words(self, verbose : bool = False):
        """perform bag of words analysis -- counts the frequency of each word in the text"""
        bag = {}

        for word in self.text.split():
            if word.lower() in bag and word.lower() not in ENGLISH_STOP_WORDS:  # exclude stop words from the bag
                bag[word.lower()] += 1
            else:
                bag[word.lower()] = 1

        if verbose:
            tmp = sorted(bag.items(), key=lambda item: item[1], reverse=True)
            print("top 5 most frequent words:")
            print(tmp[:5])
            print("top 5 least frequent words:")
            print(tmp[-5:])

        self.analysis_results['bag of words'] = bag

    def text_length(self):
        """calculates the number of words and sentences in the text"""
        pass

    def part_of_speech_tagging(self, verbose : bool = False):
        """identifies nouns, verbs, and adjectives in the text"""
        tagged = []
        # sent_tokenize is one of instances of PunktSentenceTokenizer from the nltk.tokenize.punkt module
        tokenized = sent_tokenize(self.text)
        for i in tokenized:
            # Word tokenizers is used to find the words and punctuation in a string
            wordsList = nltk.word_tokenize(i)

            # removing stop words from wordList
            wordsList = [w for w in wordsList if not w in ENGLISH_STOP_WORDS]

            tagged = nltk.pos_tag(wordsList)
            if verbose:
                print(tagged)

        # Go through and count the occurences of each POS
        res = {}
        for word, pos in tagged:
            if pos in res:
                res[pos] += 1
            else:
                res[pos] = 1

        if verbose:
            for key, value in res.items():
                print(f"{key} : {value}")

        self.analysis_results['part of speech tagging'] = res


    def first_person_statements(self):
        """identifies and counts the number of 'I' statements in the text"""
        pass

    def semantic_analysis(self) -> None:
        """perform semetic analysis"""
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        inputs = tokenizer(self.text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()

        self.analysis_results["verbose semantic analysis"] = \
            {x: y for x, y in zip(["NEGATIVE", "POSITIVE"], logits.softmax(-1).tolist()[0])}
        self.analysis_results["semantic classification"] = model.config.id2label[predicted_class_id]

    def export_data(self):
        """Function that exports the data"""
        pass


if __name__ == "__main__":
    model = ExploratoryAnalysis(
        "In parallel, and in keeping with previous fndings in similarrepeated decision-making tasks, chronological age was associated with increasing noisiness of choices relative tovalues estimated using standard reinforcement learning, and a concurrent increase in perseverative responding. In Experiment 2, we delved further into the relationship between memory precision and choice, by identifying a role for memory precision in selecting which memories are sampled. Specifcally, we designed a variant of the previous task in which sampled context memories could be identifed as specifc or ‘gist’-level (e.g. ‘beaches’ as opposed to ‘that one particular beach’), with each having distinct, opposing efects on choice. We found that lower memory precision was associated with a greater reliance on gist-based memory during memory sampling"
    )
    model.part_of_speech_tagging(True)
