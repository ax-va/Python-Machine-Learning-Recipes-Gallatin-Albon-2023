"""
Create a set of features indicating the number of times an observation’s text contains a particular word.
We can set every feature to be the combination of two words (called a 2-gram) or even three words (3-gram).

See also:

Bag of Words Meets Bags of Popcorn
https://www.kaggle.com/c/word2vec-nlp-tutorial
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

text_data = np.array(
    [
        'I love Brazil. Brazil!',
        'Sweden is best',
        'Germany beats both',
    ]
)

# Create the "bag of words" feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
# <3x8 sparse matrix of type '<class 'numpy.int64'>'
#         with 8 stored elements in Compressed Sparse Row format>

bag_of_words = bag_of_words.toarray()
# array([[0, 0, 0, 2, 0, 0, 1, 0],
#        [0, 1, 0, 0, 0, 1, 0, 1],
#        [1, 0, 1, 0, 1, 0, 0, 0]])

count.get_feature_names_out()
# array(['beats', 'best', 'both', 'brazil', 'germany', 'is', 'love',
#        'sweden'], dtype=object)

# "I" is not the array because the default token_pattern
# only considers tokens of two or more alphanumeric characters

df = pd.DataFrame(
    data=bag_of_words,
    columns=count.get_feature_names_out(),
)
#    beats  best  both  brazil  germany  is  love  sweden
# 0      0     0     0       2        0   0     1       0
# 1      0     1     0       0        0   1     0       1
# 2      1     0     1       0        1   0     0       0

# 1.
# "ngram_range" sets the minimum and maximum size of our n-grams.
# For example, (1, 3) sets all 1-grams, 2-grams, and 3-grams.

# 2.
# Remove low-information filler words by using "stop_words",
# either with a built-in list or a custom list.

# 3.
# Restrict the words or phrases we want to consider to a certain list of words using "vocabulary".

count_2gram = CountVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    vocabulary=['brazil', 'sweden', 'germany', 'love', 'love brazil'])
bag = count_2gram.fit_transform(text_data)
bag.toarray()
# array([[2, 0, 0, 1, 1],
#        [0, 1, 0, 0, 0],
#        [0, 0, 1, 0, 0]])

# View the 1-grams and 2-grams
count_2gram.vocabulary_
# {'brazil': 0, 'sweden': 1, 'germany': 2, 'love': 3, 'love brazil': 4}
