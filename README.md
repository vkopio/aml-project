# aml-project

The Advanced Machine Learning course project.

## File structure

The following table constains descriptions about the files:

| **File** | **Description** |
| -------- | --------------- |
| `analysis.py` | Data exploration. |
| `common.py` | Common utils for data loading etc. |
| `lda_parameters.py` | Scripts for determining best hyperparameters for LDA. |
| `model_*.py` | Model cross validation and training. |
| `preprocess_lda.py` | Scripts for creating LDA train and test files. |

## Running LDA preprocessing

The `mallet` adapter has been removed from the newer versions of `gensim`. To to use the adapter, version `3.8.3` needs to be installed:

    pip install gensim==3.8.3
