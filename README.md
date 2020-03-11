# Nintent

**Nintent** is a research project using deep learning combined with a top-down parsing algorithm to parse nested queries. Example applications include parsing complex user queries for smart assistants.


# Detailed paper

More in-depth informations about the project can be found in the project report. Check the `report.md` file or generate the pdf version using `./make_report.sh` (to do so, you need pandoc and a latex distribution installed). You can also download the report directly from [here](https://drive.google.com/open?id=1UbUyf57dc8fzP_Z-UZKmebMoGD3Ei9tc).


# Training a model

## Prerequisites

### Getting the dataset

We use a dataset from *Gupta et al., 2018* consisting of about 45000 user queries.

* Download the dataset here : http://fb.me/semanticparsingdialog 
* unzip it
* place every file under the main directory under "datas"


### Getting SpanBERT

The `setup.sh` script will download SpanBERT and place it under the *models* 
directory for you.


## Training

The `train.py` script can be used to train a model. Running `python3 train.py` will train a model with the default training config (see `configs/default-train-config.json`). Use `python3 train.py --help` for more informations.


## Scoring 

Run the `score.py` script to score a previously saved model. The _--model-path_ argument can be specified to score a specific model. Refer to `python3 score.py --help` for more informations.
