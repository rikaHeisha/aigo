
## Training a model

### Setup
First download all pip packages:
```
pip install hydra-core --upgrade
pip install torch numpy fire matplotlib plotly
pip install ai_edge_torch
pip install -U kaleido

```

Before training a model, you need to setup your shell environment. This only needs to be done once whenever you create a new shell. Run:
```
source tools/setup.sh
```

Note:
Run this to see all the pip packages installed
```
pip3 freeze
```
 

### Training


##  Tensorboad
Run the following command to launch tensorboard:
```
tensorboard --logdir=/home/rmenon/Desktop/dev/ml_results/aigo_results
```

## Jupyter Notebook
Run the following to launch a jupyter notebook:
```
jupyter-notebook --dir=<dir>
```

## Unit tests
Unit tests are run using pytest. pytest automatically finds all python files `*_test.py` and execute any function `test_*()` defined in those files. To run all unit tests:

```
# Run all unit tests
python go_detection/pytest_wrapper.py

# Show stdout even on success
python go_detection/pytest_wrapper.py -rP

# Run a specific function
python go_detection/pytest_wrapper.py -rP -k <function_name>

```