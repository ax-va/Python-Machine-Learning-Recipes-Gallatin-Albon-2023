"""
popular sample datasets in scikit-learn:
- load_iris with 150 observations on the measurements of iris flowers for exploring classification algorithms
- load_digits with 1,797 observations from images of handwritten digits for teaching image classification
"""
from sklearn import datasets

# Load digits dataset
digits = datasets.load_digits()

# features matrix
features = digits.data
# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#        [ 0.,  0., 10., ..., 12.,  1.,  0.]])

features[0]
# array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
#        15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
#        12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
#         0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
#        10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])

features.shape
# (1797, 64)

# target vector
target = digits.target
# array([0, 1, 2, ..., 8, 9, 8])

target.shape
# (1797,)

print(digits.DESCR)
# .. _digits_dataset:
#
# Optical recognition of handwritten digits dataset
# --------------------------------------------------
#
# **Data Set Characteristics:**
#
# :Number of Instances: 1797
# :Number of Attributes: 64
# :Attribute Information: 8x8 image of integer pixels in the range 0..16.
# :Missing Attribute Values: None
# :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
# :Date: July; 1998
#
# This is a copy of the test set of the UCI ML hand-written digits datasets
# https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
#
# The data set contains images of hand-written digits: 10 classes where
# each class refers to a digit.
#
# Preprocessing programs made available by NIST were used to extract
# normalized bitmaps of handwritten digits from a preprinted form. From a
# total of 43 people, 30 contributed to the training set and different 13
# to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
# 4x4 and the number of on pixels are counted in each block. This generates
# an input matrix of 8x8 where each element is an integer in the range
# 0..16. This reduces dimensionality and gives invariance to small
# distortions.
#
# For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
# T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
# L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
# 1994.
#
# |details-start|
# **References**
# |details-split|
#
# - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
#   Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
#   Graduate Studies in Science and Engineering, Bogazici University.
# - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
# - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
#   Linear dimensionalityreduction using relevance weighted LDA. School of
#   Electrical and Electronic Engineering Nanyang Technological University.
#   2005.
# - Claudio Gentile. A New Approximate Maximal Margin Classification
#   Algorithm. NIPS. 2000.
#
# |details-end|
