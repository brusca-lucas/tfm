# Forex Prediction with Deep Learning

![status](https://img.shields.io/badge/status-in%20progress-yellow)  

This project is currently **in progress** and not yet production-ready.  
Expect breaking changes and incomplete features.  

This repository contains the development of a deep learning model for Forex prediction.
The aim is to explore time series modeling and evaluate models not only by accuracy but also by financial performance.

## Objectives

Build a deep learning model for Forex trading signals.

Keep a modular and reusable codebase for experimentation.

Compare modeling approaches and loss functions for time series.

Evaluate models with both classification metrics and profitability metrics.

Prepare the system for possible production deployment.

## Project Structure
```
.
├── models
│   ├── keras
│   └── tree_based
├── notebooks
├── src
│   ├── utils
│   │   ├── eda_utils.py
│   │   └── general_utils.py
│   ├── data_building.py
│   ├── evaluation_and_metrics.py
│   └── model.py
├── README.md
├── config.yaml
├── directr.txt
├── poetry.lock
└── pyproject.toml
```
## Setup

This project uses Poetry for dependency management and pyenv for Python version control.
```
Clone the repository:

git clone https://github.com/brusca-lucas/tfm.git
cd tfm
```

Set the Python version (via pyenv):
```
pyenv install 3.x.x
pyenv local 3.x.x
```

Install dependencies with Poetry:
```
poetry install
```

Activate the environment:
```
poetry shell
```
