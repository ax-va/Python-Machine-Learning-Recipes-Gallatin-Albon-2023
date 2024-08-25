Serve a trained TensorFlow model using a web server.

->

Use the open source TensorFlow Serving framework and Docker.

```unix
sudo docker run -p 8515:8515 \
--mount type=bind,source=$(pwd)/models/saved_model,target=/models/saved_model/1 \
-e MODEL_NAME=saved_model \
-t tensorflow/serving
```