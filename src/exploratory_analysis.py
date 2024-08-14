from transformers import pipeline

class ExploratoryAnalysis:
    def __init__(self, intial_description : str, memorable_description : str):
        self.initial_description = intial_description
        self.memorable_description = memorable_description

        # dicts that are updated in every analysis technique
        self.initial_dict = {}
        self.memorable_dict = {}


    def create_report(self):
        """Function that calls performs analysis methods and compiles 2 dictionaries corresponding to the initial
        description and memorable description"""
        pass


    def export_data(self):
        """Function that exports the data"""
        pass


if __name__ == "__main__":
    pass
