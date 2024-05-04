"""
Stem words = convert words into their root forms.

Stemmed words we are less readable to humans but closer to its base meaning
and therefore more suitable for comparison across observations.

NLTK's PorterStemmer implements the widely used Porter stemming algorithm:
https://tartarus.org/martin/PorterStemmer/
"""
from nltk.stem.porter import PorterStemmer

# Create word tokens
tokenized_words = ['i', 'am', 'is', 'humbled', 'by', 'this', 'tradition', 'traditional', 'meeting']

# Create stemmer
porter = PorterStemmer()

# Apply stemmer
[porter.stem(word) for word in tokenized_words]
# ['i', 'am', 'is', 'humbl', 'by', 'thi', 'tradit', 'tradit', 'meet']
