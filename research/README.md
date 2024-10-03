
## Training a model

### Setup
Before training a model, you need to setup your shell environment. This only needs to be done once whenever you create a new shell. Run:
```
source tools/setup.sh
```

### Training


##  Tensorboad
Run the following command to launch tensorboard:
```
tensorboard --logdir=/home/rmenon/Desktop/dev/ml_results
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