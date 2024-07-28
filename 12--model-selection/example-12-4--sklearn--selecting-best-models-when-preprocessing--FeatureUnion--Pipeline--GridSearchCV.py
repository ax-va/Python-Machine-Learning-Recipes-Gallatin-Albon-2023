"""
Include a preprocessing step during model selection.
->
Create a pipeline that includes the preprocessing step
and any of its parameters to use in GridSearchCV.

Notice that preprocessing should be built into the pipeline.

Explanation:
The general strategy is preprocessing only the training data and
then applying that transformation to the test data.
Thus, the test data know nothing about the training data.
In the k-fold cross-validation, that can only be done correctly by using a pipeline.
If we preprocess all the data before the k-fold cross-validation,
the training and test data are not isolated from each other and have exchanged information.
See also an experiment below.
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(0)

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

features.shape
# (150, 4)

target.shape
# (150,)

# Combine two preprocessing steps:
# 1. standardize the feature values (StandardScaler);
# 2. principal component analysis (PCA).
preprocess = FeatureUnion(
    [
        ("std", StandardScaler()),
        ("pca", PCA()),
    ]
)

# Create a pipeline
pipe = Pipeline(
    [
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, solver='liblinear')),
    ]
)

# Create space of candidate values
search_space = [
    {
        "preprocess__pca__n_components": [1, 2, 3],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": np.logspace(0, 4, 10),
    }
]

# Create grid search
grid_search = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
# Fit grid search
best_model = grid_search.fit(features, target)
# best model
best_model.best_estimator_
# Pipeline(steps=[('preprocess',
#                  FeatureUnion(transformer_list=[('std', StandardScaler()),
#                                                 ('pca', PCA(n_components=1))])),
#                 ('classifier',
#                  LogisticRegression(C=7.742636826811269, max_iter=1000,
#                                     penalty='l1', solver='liblinear'))])

# best n_components
best_model.best_estimator_.get_params()['preprocess__pca__n_components']
# 1

# # experiment: preprocessing is not built into the pipeline -> worse "best" model

features_std = StandardScaler().fit_transform(features)
features_pca = PCA(n_components=1).fit_transform(features_std)

pipe_v2 = Pipeline(
    [
        ("classifier", LogisticRegression(max_iter=1000, solver='liblinear')),
    ]
)

search_space_v2 = [
    {
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": np.logspace(0, 4, 10),
    }
]

grid_search_v2 = GridSearchCV(pipe_v2, search_space_v2, cv=5, verbose=0, n_jobs=-1)
best_model_v2 = grid_search_v2.fit(features_pca, target)
best_model_v2.best_estimator_
# Pipeline(steps=[('classifier',
#                  LogisticRegression(C=2.7825594022071245, max_iter=1000,
#                                     penalty='l1', solver='liblinear'))])
