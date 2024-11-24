from transformers import AutoTokenizer
from icecream import ic

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

ic(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

ic(ids)
