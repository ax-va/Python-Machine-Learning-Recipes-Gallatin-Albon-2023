"""
Obtain a bag of words with words weighted by their importance to a document with TF-IDF.
->
The higher the value, the more important the word is to a document.

TF-IDF = Term Frequency - Inverse Document Frequency

The more a word appears in a document, the more likely
it is that the word is important to that document
->
term frequency

If a word appears in many documents, it is likely less
important to any individual document
->
document frequency

A score to every word representing how important that word is in a document:
tf‐idf(t, d) = tf(t, d) * idf(t),
where t is a term (word), d is a document, and idf is the inverse of document frequency.

In scikit-learn:
idf(t) = log((1 + n_d) / (1 + tf(t, d))) + 1,
where n_d is the number of documents and tf(t, d) the number of documents where the term appears.

Then, scikit-learn normalizes the tf‐idf vectors using the Euclidean norm (L2 norm).

See also:
https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = np.array(
    [
        'I love Brazil. Brazil!',
        'Sweden is best',
        'Germany beats both',
    ]
)

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# <3x8 sparse matrix of type '<class 'numpy.float64'>'
#         with 8 stored elements in Compressed Sparse Row format>

# Show tf-idf feature matrix as dense matrix
feature_matrix = feature_matrix.toarray()
# array([[0.        , 0.        , 0.        , 0.89442719, 0.        ,
#         0.        , 0.4472136 , 0.        ],
#        [0.        , 0.57735027, 0.        , 0.        , 0.        ,
#         0.57735027, 0.        , 0.57735027],
#        [0.57735027, 0.        , 0.57735027, 0.        , 0.57735027,
#         0.        , 0.        , 0.        ]])

tfidf.get_feature_names_out()
# array(['beats', 'best', 'both', 'brazil', 'germany', 'is', 'love',
#        'sweden'], dtype=object)

# Show feature names
tfidf.vocabulary_
# {'love': 6,
#  'brazil': 3,
#  'sweden': 7,
#  'is': 5,
#  'best': 1,
#  'germany': 4,
#  'beats': 0,
#  'both': 2}

df = pd.DataFrame(
    data=feature_matrix,
    columns=tfidf.vocabulary_,
)
#       love   brazil   sweden        is     best  germany     beats     both
# 0  0.00000  0.00000  0.00000  0.894427  0.00000  0.00000  0.447214  0.00000
# 1  0.00000  0.57735  0.00000  0.000000  0.00000  0.57735  0.000000  0.57735
# 2  0.57735  0.00000  0.57735  0.000000  0.57735  0.00000  0.000000  0.00000

# The higher the value, the more important the word is to a document
