"""
Train a naive Bayes classifier for discrete or count data.
->
Use a multinomial naive Bayes classifier.

One of the most common uses of multinomial naive Bayes
is text classification using bags of words or TF-IDF.
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array(
    [
        'I love Brazil. Brazil!',
        'Brazil is best',
        'Germany beats both',
    ]
)

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
# <3x7 sparse matrix of type '<class 'numpy.int64'>'
#         with 8 stored elements in Compressed Sparse Row format>

# column names of the matrix below
count.get_feature_names_out()
# array(['beats', 'best', 'both', 'brazil', 'germany', 'is', 'love'],
#       dtype=object)

# Create feature matrix
features = bag_of_words.toarray()
# array([[0, 0, 0, 2, 0, 0, 1],
#        [0, 1, 0, 1, 0, 1, 0],
#        [1, 0, 1, 0, 1, 0, 0]])

# Create target vector
target = np.array([0, 0, 1])

# # Create multinomial naive Bayes with prior probabilities of each class
classifier = MultinomialNB(
    # to set priors for two classes (brazil and germany);
    # if not set, they are learned from data
    class_prior=[0.25, 0.5],
    # # to use a uniform distribution for classes
    # class_prior = None,
    # fit_prior=False,
    # # smoothing hyperparameter; "alpha=0.0" -> no smoothing
    # alpha=1.0,
)
# Train model
model = classifier.fit(features, target)

# Create new observations
new_observations = [
    [0, 0, 0, 1, 0, 1, 0],  # brazil
    [0, 0, 0, 0, 1, 1, 0],  # germany
    [0, 0, 0, 1, 1, 1, 0],  # brazil and germany
]
# Predict new observations' classes
model.predict(new_observations)
# array([0, 1, 1])  # -> brazil, germany, germany
