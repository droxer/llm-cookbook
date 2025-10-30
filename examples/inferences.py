from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from rich import print as rprint

model_name = "google/gemma-3-270m" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,   # we avoid bitsandbytes on Mac
    torch_dtype="auto",   # let PyTorch decide (float32 or float16 if MPS allows)
    device_map={"": "mps"}  # this places the model on the Apple GPU (MPS). 
    # If MPS is not available or if issues occur, use {"": "cpu"} to train on CPU.
)

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("mps")

# past_key_values = DynamicCache()
generated_ids = model.generate(**model_inputs, max_length=30, cache_implementation="dynamic")
rprint(tokenizer.batch_decode(generated_ids)[0])
# '<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'

