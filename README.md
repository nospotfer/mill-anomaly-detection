
[![PythonVersion](https://img.shields.io/pypi/pyversions/gino_admin)](https://img.shields.io/pypi/pyversions/gino_admin)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# `PAI Vertical Roller Mill - Anomaly Detection`

> project for: `Insus.ch`  

## Objectives and Requirements


* Identify any anomalies in the data using standard packages and frameworks in Python (e.g. using specific autoencoders or isolation forest).
* Write your code with reproducibility in mind (the concept of the `reproducibility of the research`) which you will share as one of the results.
* Report your experiments, including the elements you included or excluded and why, results, and metrics used.
* Make your code testable by creating unit testing.


# Configuration, Usage and Deployment

## Installing the package

1. Clone the repo

    ```bash
    git git@github.com:nospotfer/mill-anomaly-detection.git
    cd mill-anomaly-detection
    ```

2. Install dependencies using [pip](https://pip.pypa.io/en/stable/installing/). The following command
will install the dependencies from `setup.py`. In the backend it will run `pip install -e ".[test, serve]"`. Note that installing dependencies with `-e` 
editable mode is needed to properly run unit tests. `[test, serve]` is optional. `test` refers to
unit test dependencies and `serve` refers to deployment dependencies.

    ```bash
    make install
    ```

## Running the project

Preferably, you can use make commands (from `Makefile`) or directly run scripts from `scripts`.  
Refer to section below for the descriptions of make commands. Before running it, consider creating  
a virtual environment.  

**Makefile and test example**

Try out the `make` commands on the CSV file `df_poc.csv` dataset model (see `make help`).

```
clean                          clean artifacts
coverage                       create coverage report
generate-dataset               run ETL pipeline
help                           show help on available commands
lint                           flake8 linting and black code style
run-pipeline                   clean artifacts -> generate dataset -> train -> serve
serve                          serve trained model with a REST API using dploy-kickstart
test-docker                    run unit tests in docker environment
test                           run unit tests in the current virtual environment
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
```

Note the dependency: `generate-dataset` > `train` > `serve`.

## Docker

Currently, you can find the following docker files:  
1. `jupyter.Dockerfile` builds an image for running notebooks.  
2. `test.Dockerfile` builds an image to run all tests in (`make test-docker`).
3. `serve.Dockerfile` build an image to serve the trained model via a REST api.
To ease the serving it uses open source `dploy-kickstart` module. To find more info
about `dploy-kickstart` click [here](https://github.com/dploy-ai/dploy-kickstart/).

Finally, you can start all services using `docker-compose`:  
for example `docker-compose up jupyter` or `docker-compose up serve`.  

Do you need a notebook for development? Just run `docker-compose up jupyter`. It will launch a Jupyter Notebook 
with access to your local development files.

## Deploying the API

Calling `make serve` will start a Flask based API using `dploy-kickstart`
wrapper. 

In `ml_core/model/predict.py` file, there is `# @dploy endpoint predict`
annotation above the `predict` method. 

From `# @dploy endpoint predict` annotation, we are telling `dploy-kickstart` 
that the url that we need to do the post request is `http://localhost:8080/predict`.
As another example, if the annotation would be `# @dploy endpoint score` then the url
would change to `http://localhost:8080/score`.  

Going back to our case, the posted data to `http://localhost:8080/predict` url will be
the argument of the exposed method which is `def predict(body)`. 

As a concrete example;

After calling `make serve`, we can do our predictions with the following curl command.
In this case, `def predict(body)` method will be triggered and the value of the `--data`
will be the argument of `def predict(body)` function, i.e. `body`.

```sh
 curl --request POST \
  --url http://localhost:8080/predict \
  --header 'content-type: application/json' \
  --data '{"model_f_name": "lr.joblib",
           "features": [0,2.390000105,1.5,2.109999921,0.839999974,0.17,0.150000006,1.32,0.93,0.200000003,18.5,37.9,47.125,0.282499999,0.189999998,0.449999988,1.7,2.205,488.3808406,31.4554075,10.9313475,58.1055925,69.857555,63.3372025,145,144.2604,143.2838,144.2231,144.72035,143.697825,37.76956044,48.14815,42.36111,55.61343,56.42360765,47.85879637,50.40509033,83.73842,68.11343,77.25694069,72.68518,71.18055725,59.08564758,58,95,94,50,45,52,51,52,51,0.6799584,0.5902618,38,48.79999924,58.5,58.90000153,59.825,48,39.86437,192.432725,74.14704,11.0450725,9.9954285,43.60000229,57.60000229,65.52499962,78.92500114,83.90000153,100.5,57.79999943,57.90000153,60,8.03,8.279999733,7.320000172,7.619999886,6.7725,6.682,6.89925,6.842,5.01825,4.885,4.8055,4.57225,4.22775,4.22775,501.55525,545.912925,1.403356,63.2595475,48.556855,39.6050375,1.099537,74.86979,83.315245,75.647425,74.65278,72.193285,82.4833625,72.627315,77.5643825,70.457175,73.441115,70.52951,78.848375,6.860532,83.5]
           }'
```

To test the health of the deployed model, you can make a get request as shown below;

```sh
    curl --request GET \
      --url http://localhost:8080/healthz
```



## Project Structure Overview 
The project structure tree is shown below:

```
.
├── .github             # Github actions CI pipeline
|
├── data                
│   ├── predictions     # predictions data, calculated using the model
│   ├── raw             # immutable original data
│   ├── staging         # data obtained after preprocessing, i.e. cleaning, merging, filtering etc.
│   └── transformed     # data ready for modeling (dataset containing features and label)
|
├── docker              # Store all dockerfiles
|
├── ml_core      # Logic of the model
│   ├── etl             # Logic for cleaning the data and preparing train / test set 
│   └── model           # Logic for ML model including CV, parameter tuning, model evaluation
|
├── models              # Store serialized fitted models
|
├── notebooks           # Store prototype or exploration related .ipynb notebooks. Use it locally. Avoid adding jupyter
│                       # notebooks to git as a best practice unless required..
|
├── reports             # Contains textual or visualisation reports
|
├── scripts             # Call ml_core module from here e.g. cli for training
|
└── tests               # Unit tests
```

## Best practices

- Make sure that `docker-compose up test` runs properly.  
- In need for a Notebook? Use the docker image: `docker-compose up jupyter`.
- Commit often, perfect later.
- Integrate `make test` with your CI pipeline.
- Capture `stdout` when deployed.


> copyright by `Gabriel Oliveira-Barra`
> main developer `Gabriel Oliveira-Barra` (`gabriel@oliveira-barra.com`)
