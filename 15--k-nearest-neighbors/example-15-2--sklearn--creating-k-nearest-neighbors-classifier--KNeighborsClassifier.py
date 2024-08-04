"""
Predict a class of an observation based on the class of its neighbors.
->
If the dataset is not very large, use Scikit-Learn's KNeighborsClassifier.

The probability of an observation $x_u$ to be class $j$ using $k$ nearest neighbors:
$$
P_j = 1 / k \sum_{i in \I_u} 1(y_i = j)
$$,
where $I_u$ is the index neighborhood of $x_u$,
$y_i$ is the known class of the i-th observation, and
$1$ is the indicator function (equals to 1 if the argument is true and 0 if false).
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# When using any learning algorithm based on distance, it is important
# to transform features so that they are on the same scale.
standardizer = StandardScaler()

# Standardize features
X_std = standardizer.fit_transform(X)

# Train a KNN classifier with 5 neighbors
knn = KNeighborsClassifier(
    # metric="euclidean"  # by default
    n_neighbors=5,
    n_jobs=-1,  # Using multiple cores is highly recommended
    # weights="distance",  # ->
    # # The closer observations' votes are weighted more than observations farther away.
    # # This make sense, since more similar neighbors give us more useful information.
).fit(X_std, y)

# Create new observations
new_observations = [[.75, .75, .75, .75],
                    [.5, .5, .5, .5],
                    [1., 1., 1., 1.]]

# Predict the class of two observations
knn.predict(new_observations)
# array([1, 1, 2])

# Get probability that each observation is one of three iris' classes
knn.predict_proba(new_observations)
# array([[0. , 0.6, 0.4],
#        [0. , 1. , 0. ],
#        [0. , 0. , 1. ]])
