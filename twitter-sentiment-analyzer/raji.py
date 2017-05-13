import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

text = "HI How are you? I am fine and uoui"
token = nltk.word_tokenize(text)
bigrams = ngrams(token,2)

print Counter(bigrams)


import nltk
from nltk import word_tokenize
from nltk.util import bigrams
from collections import Counter

text = "HI How are you? I am fine and uoui"
token = nltk.word_tokenize(text)
bigram = nltk.bigrams(token)

print Counter(bigram)
