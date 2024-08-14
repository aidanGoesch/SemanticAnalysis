import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class ExploratoryAnalysis:
    def __init__(self, text : str):
        self.text = text

        # dict that is updated in every analysis member function
        self.analysis_results = {}


    def create_report(self):
        """Function that calls performs analysis methods and compiles 2 dictionaries corresponding to the initial
        description and memorable description"""
        pass

    def semantic_analysis(self) -> None:
        """perform semetic analysis on the member variable 'text'"""
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
    model = ExploratoryAnalysis("Hello, my dog is gross and weird but I barely love him ")
    print(model.semantic_analysis())
