# Serving TensorFlow Models

Serve a trained TensorFlow model using a web server.
->
Use the open source TensorFlow Serving framework and Docker.

## See also:
- TensorFlow: Serving Models
https://www.tensorflow.org/tfx/guide/serving

Run a container using the public `tensorflow/serving` image
and mount the `saved_model` path of the current working directory to 
`/models/saved_model/1` inside the container.
```unix
$ sudo docker run -p 8501:8501 \
--name tf_serving \
--mount type=bind,source=$(pwd)/models/saved_model,target=/models/saved_model/1 \
-e MODEL_NAME=saved_model \
-t tensorflow/serving
```

Open `http://localhost:8501/v1/models/saved_model`
```json
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

Open `http://localhost:8501/v1/models/saved_model/metadata`
```json
{
"model_spec":{
 "name": "saved_model",
 "signature_name": "",
 "version": "1"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "__saved_model_init_op": {
   "inputs": {},
   "outputs": {
    "__saved_model_init_op": {
     "dtype": "DT_INVALID",
     "tensor_shape": {
      "dim": [],
      "unknown_rank": true
     },
     "name": "NoOp"
    }
   },
   "method_name": "",
   "defaults": {}
  },
  "serving_default": {
   "inputs": {
    "input_layer": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "10",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_input_layer:0"
    }
   },
   "outputs": {
    "output_0": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "StatefulPartitionedCall_1:0"
    }
   },
   "method_name": "tensorflow/serving/predict",
   "defaults": {}
  },
  "serve": {
   "inputs": {
    "input_layer": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "10",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serve_input_layer:0"
    }
   },
   "outputs": {
    "output_0": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "StatefulPartitionedCall:0"
    }
   },
   "method_name": "tensorflow/serving/predict",
   "defaults": {}
  }
 }
}
}
}
```

Predict a value
```ubuntu
$ curl -X POST http://localhost:8501/v1/models/saved_model:predict -d '{"inputs":[[1,2,3,4,5,6,7,8,9,10]]}'
```
```ubuntu
{
    "outputs": [
        [
            7.22400379
        ]
    ]
}
```