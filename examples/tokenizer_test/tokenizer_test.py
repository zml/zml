import argparse
import transformers


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt_path", type=str)
parser.add_argument("-m", "--model", type=str)

args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
prompt_file = open(args.prompt_path, "r")
tokens = tokenizer.encode(prompt_file.read())

print(','.join([str(t) for t in tokens]))
