"""
Evaluate the effect of the number of observations in the training set on some metric (accuracy, F_1, etc.).
->
Plot the accuracy against the training set size.

*Learning curves* visualize the dependency of the performance
(e.g., accuracy, recall) of a model on the training sets during cross-validation.
->
Determine if a learning algorithm would benefit from increasing the training data.

See also:
- Scikit-Learn: Learning curve
https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# Load data
digits = load_digits()
# Create feature matrix and target vector
features, target = digits.data, digits.target

features.shape
# (1797, 64)
target.shape
# (1797,)

# Create cross-validation training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(),  # classifier
    features,  # feature matrix
    target,  # target vector
    cv=10,  # number of folds in cross-validation
    scoring='accuracy',  # performance metric
    n_jobs=-1,  # all computer cores to use
    train_sizes=np.linspace(  # training set sizes
        start=0.01,  # ~0.9% of training data
        stop=1.0,  # ~90% of training data, ~10% of data for test
        num=50,  # different training set sizes
    )
)

np.linspace(0.01, 1.0, 50).shape
# (50,)
np.linspace(0.01, 1.0, 50)
# array([0.01      , 0.03020408, 0.05040816, 0.07061224, 0.09081633,
#        0.11102041, 0.13122449, 0.15142857, 0.17163265, 0.19183673,
#        0.21204082, 0.2322449 , 0.25244898, 0.27265306, 0.29285714,
#        0.31306122, 0.33326531, 0.35346939, 0.37367347, 0.39387755,
#        0.41408163, 0.43428571, 0.4544898 , 0.47469388, 0.49489796,
#        0.51510204, 0.53530612, 0.5555102 , 0.57571429, 0.59591837,
#        0.61612245, 0.63632653, 0.65653061, 0.67673469, 0.69693878,
#        0.71714286, 0.73734694, 0.75755102, 0.7777551 , 0.79795918,
#        0.81816327, 0.83836735, 0.85857143, 0.87877551, 0.89897959,
#        0.91918367, 0.93938776, 0.95959184, 0.97979592, 1.        ])

train_sizes.shape
# (50,)
train_sizes
# array([  16,   48,   81,  114,  146,  179,  212,  244,  277,  310,  342,
#         375,  408,  440,  473,  506,  538,  571,  604,  636,  669,  702,
#         734,  767,  800,  832,  865,  898,  930,  963,  996, 1028, 1061,
#        1094, 1126, 1159, 1192, 1224, 1257, 1290, 1322, 1355, 1388, 1420,
#        1453, 1486, 1518, 1551, 1584, 1617])

train_scores.shape
# (50, 10)
train_scores
# array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
# ...
#        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

test_scores.shape
# (50, 10)
test_scores
# array([[0.58888889, 0.61111111, 0.53333333, 0.41111111, 0.55      ,
#         0.49444444, 0.57222222, 0.51396648, 0.52513966, 0.50837989],
#        [0.75555556, 0.75      , 0.62777778, 0.52777778, 0.66111111,
# ...
#         0.95555556, 0.97222222, 0.98324022, 0.93296089, 0.92178771],
#        [0.91111111, 0.98333333, 0.93333333, 0.91666667, 0.96111111,
#         0.97222222, 0.96666667, 0.95530726, 0.93296089, 0.9273743 ]])

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)  # mean in each row: going along axis 1 for each element fixed in axis 0
train_std = np.std(train_scores, axis=1)  # std in each row: going along axis 1 for each element fixed in axis 0

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)  # mean in each row: going along axis 1 for each element fixed in axis 0
test_std = np.std(test_scores, axis=1)  # std in each row: going along axis 1 for each element fixed in axis 0

# """
# |-------> axis 1
# |  1  2
# |  4  5
# |  7  8
# V
# axis 0
# """

train_mean.shape
# (50,)
train_mean
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

train_std.shape
# (50,)
train_std
# array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

test_mean.shape
# (50,)
test_mean
# array([0.53085971, 0.68673805, 0.72565798, 0.77407511, 0.77683116,
#        0.78855059, 0.79241465, 0.79407821, 0.80909994, 0.83806331,
#        0.8525388 , 0.86925512, 0.88037865, 0.8848293 , 0.89094351,
#        0.88872439, 0.90928926, 0.91486034, 0.92265673, 0.91653631,
#        0.92933892, 0.92487896, 0.92264122, 0.92039417, 0.92930168,
#        0.92707635, 0.92874922, 0.93264432, 0.9265239 , 0.92708566,
#        0.931527  , 0.93151769, 0.9326288 , 0.93264122, 0.93210118,
#        0.93988206, 0.93876164, 0.93320608, 0.94099628, 0.93655804,
#        0.94155804, 0.9393234 , 0.94378957, 0.94324022, 0.94378026,
#        0.9471167 , 0.94712911, 0.94934823, 0.9460211 , 0.94600869])

test_std.shape
# (50,)
test_std
# array([0.05327311, 0.07816747, 0.07619737, 0.08228142, 0.08116281,
#        0.0694497 , 0.08223904, 0.07805933, 0.06619759, 0.05717438,
#        0.0670278 , 0.05302715, 0.0579582 , 0.05526011, 0.05086686,
#        0.04498165, 0.03438883, 0.02813298, 0.03037546, 0.02919199,
#        0.02289104, 0.0320476 , 0.02418423, 0.02944551, 0.02992062,
#        0.0323374 , 0.035394  , 0.02801977, 0.03209075, 0.02798954,
#        0.03152662, 0.03217326, 0.02927287, 0.02646076, 0.02748926,
#        0.02609807, 0.02992006, 0.02553329, 0.02673758, 0.02749182,
#        0.02612864, 0.02629927, 0.02774866, 0.02233972, 0.02734614,
#        0.02232795, 0.02026314, 0.02639329, 0.02336405, 0.0236189 ])

# Draw lines
plt.plot(
    train_sizes, train_mean, '--',
    color="#111111",
    label="Training score",
)
plt.plot(
    train_sizes, test_mean,
    color="#111111",
    label="Test cross-validation score",
)

# Draw bands
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    color="#DDDDDD",
)
plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    color="#DDDDDD",
)

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
# plt.show()
plt.savefig('example-11-11--visualizing-effect-of-training-set-size--sklearn--learning_curve--matplotlib--fill_between.svg')
plt.close()
