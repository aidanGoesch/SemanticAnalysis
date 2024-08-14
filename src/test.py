from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import disk_offload
import torch
import os

# PYTORCH_ENABLE_MPS_FALLBACK=1

classifier = pipeline('sentiment-analysis')

# pipeline automatically applies pre processing, model, and post processing

res = classifier("This is supposed to be a positive review. I loved it!")

print(res)


# ==

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # puts text into a format that the model can understand
#
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#
# res = classifier("This is supposed to be a positive review. I loved it!")
#
# print(res)

access_token = ""

model_name = "microsoft/Phi-3-mini-4k-instruct"

input_text = "How are"

# use hardware accelerator
if torch.backends.mps.is_built():
    print('mps is used.')
    mps_device = torch.device("mps")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, use_safetensors=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                  token=access_token,
                                  torch_dtype=torch.bfloat16,
                                  device_map = mps_device,
                                             use_safetensors=True).to("mps")

# model.to("mps")

# disk_offload(model=model, offload_dir="offload")

# generator = pipeline(model = model, tokenizer = tokenizer, task='text-generation')
torch.set_grad_enabled(False)

input_string = "Hi! Nice to meet"

# tokenize to ids
input_ids = tokenizer.encode(input_string, return_tensors="pt").to("mps")

# call model() to get logits
logits = model(input_ids).logits

# only care about the last projection in the last batch
logits = logits[-1, -1]

# softmax() to get probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# keep only the top 20
probs, ids = torch.topk(probs, 20)

# convert ids to tokens
texts = tokenizer.convert_ids_to_tokens(ids)

# print
for prob, text in zip(probs, texts):
    print(f"{prob:.4f}: \"{text}\"")

# input_ids = tokenizer(input_text, return_tensors="pt").to("mps")
#
# outputs = model.generate(
#     **input_ids,
#     do_sample=False,
#     top_k=10,
#     temperature=0.1,
#     top_p=0.95,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=5
#     )
#
# print(outputs)
# for o in outputs:
#     print(tokenizer.decode(o))

# res = generator(
#     "How are ",
#     max_length=100,
#     num_return_sequences=10,
# )