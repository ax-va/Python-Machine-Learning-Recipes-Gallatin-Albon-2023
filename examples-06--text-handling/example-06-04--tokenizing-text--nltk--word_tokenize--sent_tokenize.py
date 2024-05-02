"""
Break text up into individual words or sentences.

import nltk # nltk = Natural Language Toolkit
nltk.download('punkt')  # download the first time
"""

from nltk.tokenize import word_tokenize, sent_tokenize

string = "The science of today is the technology of tomorrow"

# Tokenize words
word_tokenize(string)
# ['The', 'science', 'of', 'today', 'is', 'the', 'technology', 'of', 'tomorrow']

string = "The science of today is the technology of tomorrow. Tomorrow is today."

# Tokenize sentences
sent_tokenize(string)
#  ['The science of today is the technology of tomorrow.', 'Tomorrow is today.']
