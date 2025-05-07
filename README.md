## Installing Dependencies
Paste `pip install -r requirements.txt` into the command line in the root of directory.

Then go to https://pytorch.org/get-started/locally/ to install the right version of PyTorch.



## Things of Note

Equation for the sequentiality value of a sentence: 

$ c(s_i, h) = -\frac{1}{|s_i|} \left( \log p_{\mathrm{LM}}(s_i \mid \mathcal{T}) - \log p_{\mathrm{LM}}(s_i \mid \mathcal{T}, s_{i-h:i-1}) \right) $

is on line `209` of `src/sequentiality.py`. The sequentiality values for each sentence of a story are then averaged to get the value for the whole text (in accordance with the paper) in the method `SequentialityModel.calculate_text_sequentiality` on line `276`. The topic is set for every story using `SequentialityModel.set_topic` every time `SequentialityModel.calculate_text_sequentiality` is called

The function that divides each story into bins is on line `91` of `verification/generate_plots.py`. Code that generates plots seen in Figure 2a and 2d are in the same file.


To run the code, run use `python3 main.py` and change the function at the bottom of the file. The function that was used to generate data was `run_sequential` (definition on line `357` of `main.py`). 


