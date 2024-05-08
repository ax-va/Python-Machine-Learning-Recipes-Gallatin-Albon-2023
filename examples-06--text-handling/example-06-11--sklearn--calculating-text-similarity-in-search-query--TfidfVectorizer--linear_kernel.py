"""
Implement a text search function with TF-IDF vectors and cosine similarities.

Cosine similarities take on the range of [0, 1],
with 0 being least similar and 1 being most similar.

See also:

Cosine similarities
https://www.geeksforgeeks.org/cosine-similarity/

Nvidia gave me a $15K Data Science Workstation — here’s what I did with it
https://towardsdatascience.com/nvidia-gave-me-a-15k-data-science-workstation-heres-what-i-did-with-it-70cfb069fc35
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

text_data = np.array(
    [
        'I love Brazil. Brazil!',
        'Sweden is best',
        'Germany beats both',
    ]
)

tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# <3x8 sparse matrix of type '<class 'numpy.float64'>'
#         with 8 stored elements in Compressed Sparse Row format>

# Create a search query and transform it into a tf-idf vector
text = "Brazil is the best"
vector = tfidf.transform([text])
# <1x8 sparse matrix of type '<class 'numpy.float64'>'
#         with 3 stored elements in Compressed Sparse Row format>

linear_kernel(vector, feature_matrix)
# array([[0.51639778, 0.66666667, 0.        ]])

# Calculate the cosine similarities between the input vector and all other vectors
cosine_similarities = linear_kernel(vector, feature_matrix).flatten()
# array([0.51639778, 0.66666667, 0.        ])

cosine_similarities.argsort()
# array([2, 0, 1])

# Get the index of the most relevant items in order
related_doc_indices = cosine_similarities.argsort()[:-10:-1]
# array([1, 0, 2])

# Print the most similar texts to the search query along with the cosine similarity
print(
    [
        (text_data[i], cosine_similarities[i]) for i in related_doc_indices
    ]
)
# [('Sweden is best', 0.6666666666666666), ('I love Brazil. Brazil!', 0.5163977794943222), ('Germany beats both', 0.0)]
