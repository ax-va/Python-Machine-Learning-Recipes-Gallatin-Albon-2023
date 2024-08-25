"""
Serve a trained Scikit-Learn model using a web server.
->
Make a Python Flask application that loads trained models.
"""
import joblib
from flask import Flask, request

# Instantiate a flask app
app = Flask(__name__)

# Load the model from disk
model = joblib.load("trained_models/model_sklearn_1_4_1_post1.pkl")


# Create a predict route that takes JSON data,
# makes predictions, and returns them
@app.route("/predict", methods=["POST"])
def predict():
    print(request.json)
    inputs = request.json["inputs"]
    prediction = model.predict(inputs)
    return {"prediction" : prediction.tolist()}


# Run the app
if __name__ == "__main__":
    app.run()

# Make predictions to the application and get results
# by submitting data points to the endpoints using curl.
"""
$ curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{"inputs":[[5.1, 3.5, 1.4, 0.2]]}'
{"prediction":[0]}
"""
"""
 * Serving Flask app 'example-23-4--joblib--sklearn--flask--serving-scikit-learn-models'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
{'inputs': [[5.1, 3.5, 1.4, 0.2]]}
127.0.0.1 - - [25/Aug/2024 20:16:44] "POST /predict HTTP/1.1" 200 -
"""