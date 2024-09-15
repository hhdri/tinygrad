import numpy as np
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch, tqdm, colored
from extra.models.t5 import T5Embedder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

text = "translate English to German: Hello, how are you?"
tokenized = tokenizer(text, return_tensors="pt")
output = model.generate(**tokenized)
decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

encoded_pt = model.encoder(**tokenized).last_hidden_state.detach().numpy()

def load_T5(max_length:int = 12):
  tokenizer_link = "https://huggingface.co/google/flan-t5-base/resolve/main/spiece.model"
  model_link = "https://huggingface.co/google/flan-t5-base/resolve/main/model.safetensors"
  print("Init T5")
  T5 = T5Embedder(max_length, fetch(tokenizer_link))
  pt_1 = fetch(model_link)
  load_state_dict(T5.encoder, safe_load(pt_1), strict=False)
  return T5

model_tg = load_T5()

# text = "translate English to German: Hello, how is you?"
encoded_tg = model_tg(text).numpy()

np.testing.assert_allclose(encoded_pt, encoded_tg, atol=1e-4)