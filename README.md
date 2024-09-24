Infarct size quantification from TTC images
==============================

This project uses a deep learning segmentation model to quantify infarct size from TTC-stained heart slices in pig studies of ischemia/reperfusion.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
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
    │
    └──  src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
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
