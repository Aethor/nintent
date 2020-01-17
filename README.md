# Nintent

Welcome to the **Nintent** project repository !


# Prerequisites

## Getting the dataset

* Download the dataset here : http://fb.me/semanticparsingdialog 
* unzip it
* place every file under the main directory under "datas"


## Getting SpanBERT

The `setup.sh` script will download SpanBERT and place it under the *models* 
directory for you.


# Training a model

The `train.py` script can be used to train a model. Running `python3 train.py` will train a model with the default training config (see `configs/default-train-config.json`). Use `python3 train.py --help` for more informations.


# Scoring a model

Run the `score.py` script to score a previously saved model. The _--model-path_ argument can be specified to score a specific model. Refer to `python3 score.py --help` for more informations.


# Generating the report

More in-depth informations about the project can be found in the project report. Check the `report.md` file or generate the pdf version using `./make_report.sh` (to do so, you need pandoc and a latex distribution installed).