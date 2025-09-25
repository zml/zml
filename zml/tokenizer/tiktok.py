# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tiktoken>=0.10.0",
# ]
# ///

from pathlib import Path

import tiktoken
import regex

text = """
  two spaces
   three spaces
    four spaces
     five spaces
loading took {D}\\nHellow world"""

# text = (Path(__file__).parent / "bench.zig").read_text()

enc = tiktoken.get_encoding("o200k_harmony")
assert enc.decode(enc.encode("hello world")) == "hello world"


pat_str = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

t  = text
parts = []
while match := regex.search(pat_str, t):
  # print(f"word: {t[:match.start()]!r}, sep: {t[match.start():match.end()]!r}")
  if match.start() > 0 and t[:match.start()] != ' ':
    parts.append(t[:match.start()])
  if t[match.start():match.end()] != ' ':
    parts.append(t[match.start():match.end()])
  t = t[match.end():]

print(parts, len(parts))

# tokens = enc.encode(text)
# print(tokens)
# print("|".join([enc.decode([token]) for token in tokens]))

# ['\n', 'two', 'spaces', '\n', '  ', 'three', 'spaces', '\n', '   ', 'four', 'spaces', '\n', '    ', 'five', 'spaces', '\n', 'loading', 'took', ' {', 'D', '}\\', 'nHellow'] 22
