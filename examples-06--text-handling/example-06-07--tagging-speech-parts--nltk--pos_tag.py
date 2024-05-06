"""
Tag each word or character in text data with its part of speech.

NLTK uses the Penn Treebank parts for speech tags.

Examples of the Penn Treebank tags:

Tag         Part of speech
NNP         Proper noun, singular
NN          Noun, singular or mass
RB          Adverb
VBD         Verb, past tense
VBG         Verb, gerund or present participle
JJ          Adjective
PRP         Personal pronoun

NLTK also gives us the ability to train our own tagger -> a lot of overhead

Needed:

import nltk
nltk.download('averaged_perceptron_tagger')
"""
from nltk import pos_tag
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

text_data = "Chris loved outdoor running"

# Use pretrained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))
# [('Chris', 'NNP'), ('loved', 'VBD'), ('outdoor', 'RP'), ('running', 'VBG')]

# Filter words by tags
[word for word, tag in text_tagged if tag in ['NN','NNS','NNP','NNPS']]
# ['Chris']

# # # Convert tweet sentences into features for individual parts of speech

tweets = [
    "I am eating a burrito for breakfast",
    "Political science is an amazing field",
    "San Francisco is an awesome city",
]

tagged_tweets = []

# Tag each word and each tweet
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for _, tag in tweet_tag])

tagged_tweets
# [['PRP', 'VBP', 'VBG', 'DT', 'NN', 'IN', 'NN'],
#  ['JJ', 'NN', 'VBZ', 'DT', 'JJ', 'NN'],
#  ['NNP', 'NNP', 'VBZ', 'DT', 'JJ', 'NN']]

# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)
# array([[1, 1, 0, 1, 0, 1, 1, 1, 0],
#        [1, 0, 1, 1, 0, 0, 0, 0, 1],
#        [1, 0, 1, 1, 1, 0, 0, 0, 1]])

# 0 and 1 encodes the presence of tags

# Show feature names
one_hot_multi.classes_
# array(['DT', 'IN', 'JJ', 'NN', 'NNP', 'PRP', 'VBG', 'VBP', 'VBZ'],
#       dtype=object)
