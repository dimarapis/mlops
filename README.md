Repo developed for exercise section of course 02476 Machine Learning Operations
==============================
# Updated January 2023 @ the Technical University of Denmark
## Clone repo 
```shell
$ git clone https://github.com/dimarapis/mlops
```
## Installation
Generate an environment for *conda* or *venv*:
```shell
$ make create_environment
``` 
After activating this environment, install the requirements:
```shell
$ make requirements
```
Ensure project is installed as package:
```shell
$ pip install e .
```

## Exercises
## Day 1:
#### Build a simple model and a simple training loop

Train the model and visualize loss
```shell
$ python exercises/day_1/main.py train --lr 1e-2
```
See results: [Link to file](reports/figures/training_loss.png)

Evaluate model until achieving >85% accuracy
```shell
$ python exercises/day_1/main.py evaluate models/day1_best.pth
```
## Day 2:
#### Transform to cookiecutter format and run day1 exercises

Prepare data (as tensors)
```shell
$ make data
```

Train the model and visualize loss
```shell
$ make train
```

Evaluate model until achieving >85% accuracy - not with makefile
```shell
$ python src/models/predict_model.py models/day2_best.pth data/processed/test.pt 
```

Visualize features
```shell
$ python src/visualization/visualize.py
```
See results [Link to file](reports/figures/visualize.png)

Check that code is pep8 compliant
```shell
$ flake8 .
```

Fix errors with black and manually
```shell
$ black .
```

Fix typing exercise and test if type compliant
```shell
$ mypy exercises/typing_exercise.py
```

## Day 3:
#### Build dockers and config files
Training docker build
```shell
$ docker build -f trainer.dockerfile . -t trainer:latest
```
Run training in docker
```shell
$ docker run --name training trainer:latest
```

Create a config file with training parameters
<br/>
See: [Link to file](src/configs/config.yaml)

## Day 9:
### Distributed data loading

Plot data loading process using different amount of threads.
```shell
python lfw_dataset.py -get_timing 
``` 
See results: [Link to file](reports/figures/num_workers_more_aug.png)


Project Organization - from cookiecutter project
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │    └── visualize.py
    │   │
    │   └── configs  <- Configuration files
    │       └── config.yaml
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

