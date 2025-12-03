# Sequentiality

## Installing Dependencies
Paste `pip install -r requirements.txt` into the command line in the root of directory.

Then go to https://pytorch.org/get-started/locally/ to install the right version of PyTorch.

### *IMPORTANT*
DO NOT COMMIT API KEYS. This will cause problems and will not be good. to get around this use a `.env` file, which stores environment variables for the program. `src/keys.py` loads the environment variables so you don't have to worry about accidentally committing them. Below is a template that you should copy and paste into `.env` in the root of the repository. To run, you only need a HuggingFace key ([here](https://huggingface.co/settings/tokens)).
```
OPEN_AI_API_KEY="your_openai_api_key_here"
HUGGING_FACE_API_KEY="your_hugging_face_key_here"
```



## Description

### `sequentiality.py`
At a high level, the sequentiality model first takes in a given text (`SequentialityModel.calculate_text_sequentiality`). Then it creates a list of sentences (it uses a regualar expression to split the text by sentence). For each sentence, both the topic and contextual sequentiality are calculated (`SequentialityModel._calculate_topic_sequentiality` and `SequentialityModel._calculate_contextual_sequentiality` respectively). 

For each call of `SequentialityModel.calculate_text_sequentiality`, the topic is set dynamically using `SequentialityModel.set_topic`. More specifically, prompt engineering is used to instruct the model to condition every word of the following text on a particular topic. 


The equation for the sequentiality value of a sentence: 

![alt text](image.png)

The sequentiality values for each sentence of a story are then averaged to get the value for the whole text (in accordance with the paper) in the method `SequentialityModel.calculate_text_sequentiality` on line `276`. 

When it's finished running `SequentialityModel.calculate_text_sequentiality` returns a tuple containing the scalar sequentiality value for the entire text, as well as lists of topic, contextual and total sequentialities for every sentence in the text. 

### `main.py`
To run the code, run use `python3 main.py` and change the function at the bottom of the file. The function that was used to generate data was `run_sequential` (definition on line `357` of `main.py`). Calling `generate_plots` in `main.py` will generate Figure 2a and 2d from the original paper. 


## Output Structure

The outputs are stored in the `outputs/` folder. Each model has its own sub-folder, and within each model folder there are 9 folders numbered according to the history length used to calculate the sequentiality values in that folder. Each history length folder contains a `main.csv` file which contains the sequentiality values for the entire dataset using that model and history length. There are also smaller subsets of `main.csv` such as `main-mini.csv`, which was used to generate the plots in the paper. Please select a history length that works for you, and adhere to the data storage conventions.

## Data Structure
The dataset used in the original paper ([Hippocorpus](https://huggingface.co/datasets/allenai/hippocorpus)), is in `datasets/hippocorpus/`. There are other datasets in the other folders in `datasets/` that you can use.
