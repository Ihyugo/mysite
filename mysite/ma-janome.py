from janome.tokenizer import Tokenizer
import sys

t = Tokenizer()
inputline = sys.stdin.readline()

malist = t.tokenize(inputline)

for n in malist:
	print(n)
