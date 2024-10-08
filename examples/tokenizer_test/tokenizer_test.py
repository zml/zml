import transformers

model = "codellama/CodeLlama-7b-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(model)
tokens = tokenizer.encode("Hello world")

print(','.join([str(t) for t in tokens]))
