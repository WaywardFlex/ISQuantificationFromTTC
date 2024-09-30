Infarct size quantification from TTC images
==============================

This project uses a deep learning segmentation model to quantify infarct size from TTC-stained heart slices in pig studies of ischemia/reperfusion. Please find the original publication at: https://link.springer.com/article/10.1007/s00395-024-01081-x.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- processed, canonical data sets for modeling, to be added.
    │   └── raw            <- The original, immutable data dump, to be added.
    │
    ├── models             <- Trained and serialized models, to be added
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials, to be added.
    │
    ├── reports            <- Generated analysis, link to publication.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └──  src                <- Source code for use in this project.
        ├── __init__.py     <- Makes src a Python module
        │
        ├── train          <- Script to download, preprocess, create the data set, train, cross-validate, evaluate and visualize from training data
        │   └── train_and_evaluate.py
        │    
        ├── test           <- Script to download, preprocess, create the data set, train, evaluate and visualize from testing data
        │   └── test_and_evaluate.py
        │
        └── test_rat       <- Script to download, preprocess, create the data set, train, evaluate and visualize from testing data from rats
            └── test_on_rats.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
