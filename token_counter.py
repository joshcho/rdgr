import sys
import tiktoken

# To get the tokeniser corresponding to a specific model in the OpenAI API:
text = sys.argv[1]
model = sys.argv[2]
enc = tiktoken.encoding_for_model(model)

print(len(enc.encode(text)))
