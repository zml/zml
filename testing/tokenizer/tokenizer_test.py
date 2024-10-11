import argparse
from tokenizers import Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt_path", type=str)
parser.add_argument("-m", "--model", type=str)

args = parser.parse_args()

tokenizer = Tokenizer.from_pretrained(args.model)
prompt_file = open(args.prompt_path, "r")
tokens = tokenizer.encode(prompt_file.read()).ids

print(','.join([str(t) for t in tokens]))
