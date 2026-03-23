import transformers
from transformers import TextStreamer

model_path = "/var/models/Qwen/Qwen3.5-0.8B"
 
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={}, 
    device="cuda",
)

model, tokenizer = pipeline.model, pipeline.tokenizer

prompt = "Tell a story about a cat"

inputs = tokenizer(prompt, return_tensors="pt",add_special_tokens=True)

input_ids = inputs["input_ids"]
decoded = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)
print("raw prompt:", decoded)

streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)
pipeline(prompt, streamer=streamer, max_new_tokens=4000,top_k=1,do_sample=False)
 
