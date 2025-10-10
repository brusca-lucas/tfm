from datetime import datetime
import yaml
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from src.data_building import (add_macro,
                               add_technicals,
                               target_building_long_short)
from src.evaluation_and_metrics import print_pr_roc, equity_curve
from src.utils.general_utils import find_latest_model, sequences_generation

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class KerasModel:

    def __init__(
        self,
        layers: list,
        data: pd.DataFrame,
        interval: int,
        change: float,
        features: list,
    ):
        if layers == 'load':
            model_path = find_latest_model('keras')
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = Sequential(layers)
        self.data = data
        self.features = features
        self.interval = interval
        self.change = change

    def enrich_data(self):
        self.data = add_macro(self.data)
        self.data = add_technicals(self.data)

    def add_target(self):
        self.data = target_building_long_short(self.data,
                                               self.interval,
                                               self.change)

    def split(self, split_size: float):
        self.train_size = int(split_size * len(self.data['Target']))
        self.data_train = self.data[:self.train_size].reset_index()
        self.data_test = self.data[self.train_size:].reset_index()

    def preprocess(self, len_sequence: int):
        self.len_sequence = len_sequence
        scaler = StandardScaler()
        self.train_scaled = scaler.fit_transform(
            self.data_train[self.features]
            )
        self.train_scaled = np.column_stack(
            (self.train_scaled, self.data_train['Target'].values)
            )
        self.test_scaled = scaler.transform(
            self.data_test[self.features]
            )
        self.test_scaled = np.column_stack(
            (self.test_scaled, self.data_test['Target'].values)
            )
        self.X_train, self.y_train = sequences_generation(
            self.train_scaled,
            self.len_sequence
            )
        self.X_test, self.y_test = sequences_generation(
            self.test_scaled,
            self.len_sequence
            )

    def train(self, epochs: int, learning_rate: float):
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=learning_rate)
            )
        self.model.fit(self.X_train, self.y_train, epochs=epochs)

    def infere(self):
        self.pred_train = self.model.predict(self.X_train)
        self.pred_test = self.model.predict(self.X_test)

    def equity_curve(self, threshold: float, plot, train=False):
        if train:
            return equity_curve(
                self.pred_train,
                self.data_train["Close"],
                threshold,
                self.interval,
                plot=plot
            )
        else:
            return equity_curve(
                self.pred_test,
                self.data_test["Close"],
                threshold,
                self.interval,
                plot=plot
            )

    def save_model(self):
        base_path = config.get('MODELSPATH') + 'keras/'
        timestamp = datetime.now().strftime('%Y%m%d')
        model_path = f"{base_path}model_{timestamp}.h5"
        self.model.save(model_path)


class TreeBasedModel():
    def __init__(
        self,
        data: pd.DataFrame,
        model,
        interval: int,
        change: float,
        features: list
    ):
        if model == 'load':
            model_path = find_latest_model('tree_based/')
            self.model = joblib.load(model_path)
        else:
            self.model = model
        self.data = data
        self.features = features
        self.change = change
        self.interval = interval

    def enrich_data(self):
        self.data = add_macro(self.data)
        self.data = add_technicals(self.data)

    def add_target(self):
        self.data = target_building_long_short(self.data,
                                               self.interval,
                                               self.change)

    def split(self, split_size: float):
        train_size = int(split_size * len(self.data["Target"]))
        self.train_size = train_size
        X = self.data[self.features]
        y = self.data["Target"]
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def infere(self):
        self.probas_train = self.model.predict_proba(self.X_train)[:, 1]
        self.probas_test = self.model.predict_proba(self.X_test)[:, 1]

    def equity_curve(self, threshold: float, plot: bool, train=False):
        if train:
            metric = equity_curve(
                self.probas_train,
                self.X_train['Close'],
                threshold,
                plot=plot
            )
        else:
            metric = equity_curve(
                self.probas_test,
                self.X_test['Close'],
                threshold,
                plot=plot
            )
        return metric

    def pr_roc(self, plot: bool, train=False):
        if train:
            metric = print_pr_roc(self.y_train, self.probas_train, plot)
        else:
            metric = print_pr_roc(self.y_test, self.probas_test, plot)
        return metric

    def save_model(self):
        base_path = config.get('MODELSPATH') + 'tree_based'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{base_path}model_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
