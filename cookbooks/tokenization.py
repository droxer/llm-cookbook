from transformers import AutoTokenizer
from rich import print as rprint

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

rprint(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

rprint(ids)
