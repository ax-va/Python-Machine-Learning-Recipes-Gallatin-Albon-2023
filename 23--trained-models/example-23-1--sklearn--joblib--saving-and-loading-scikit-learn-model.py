"""
Save a trained Scikit-Learn model and load it elsewhere.
->
Save the model as a pickle file using joblib.

Notice:
joblib extends pickle for cases with large NumPy arrays
- a common occurrence for trained models in Scikit-Learn.
"""
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Get scikit-learn version to know what version
# of Scikit-Learn the model is serialized in.
sklearn_version = sklearn.__version__.replace(".", "_")

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create decision tree classifier object
classifier = RandomForestClassifier()
# Train model
model = classifier.fit(features, target)

# Save model as pickle file
joblib.dump(model, f"trained_models/model_sklearn_{sklearn_version}.pkl")

# Load model from file
classifier = joblib.load(f"trained_models/model_sklearn_{sklearn_version}.pkl")

# Create new observation
new_observation = [[5.2, 3.2, 1.1, 0.1]]
# Predict observation's class
prediction = classifier.predict(new_observation)
# array([0])
