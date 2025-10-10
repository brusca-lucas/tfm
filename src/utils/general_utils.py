import os

import yaml
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def find_latest_model(model_type: str) -> str:
    path = config.get('MODELSPATH') + model_type
    directory = os.path.dirname(path)
    if model_type == 'tree_based/':
        model_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    elif model_type == 'keras/':
        model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not model_files:
        raise FileNotFoundError(
               "No se encontraron modelos guardados en el directorio."
               )
    latest_model = max(
        model_files,
        key=lambda f: os.path.getmtime(os.path.join(directory, f))
        )
    return os.path.join(directory, latest_model)


def sequences_generation(data, len_sequence):
    X, y = [], []

    for i in range(len(data) - len_sequence):
        X.append(data[i: i + len_sequence])
        y.append(data[i + len_sequence, -1])
    return np.array(X), np.array(y)
