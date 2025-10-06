from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.evaluation_and_metrics import print_pr_roc, equity_curve
from src.data_building import add_macro, add_technicals, target_building_ups


class KerasModel:

    def __init__(
        self,
        layers: list,
        data: pd.DataFrame,
        interval: int,
        change: float,
        features: list,
    ):
        self.model = Sequential(layers)
        self.data = data
        self.features = features
        self.interval = interval
        self.change = change

    def enrich_data(self):
        self.data = add_macro(self.data)
        self.data = add_technicals(self.data)

    def add_target(self):
        self.data = target_building_ups(self.data, self.interval, self.change)

    def preprocess(self, len_sequence: int):
        self.len_sequence = len_sequence
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data[self.features])
        X, y = [], []

        for i in range(len(data_scaled) - len_sequence - 3):
            X.append(data_scaled[i: i + len_sequence])
            y.append(self.data["Target"].iloc[i + len_sequence])
        self.X = np.array(X)
        self.y = np.array(y)

    def split(self, split_size: float):
        train_size = int(split_size * len(self.X))
        self.train_size = train_size
        self.X_train, self.X_test = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]

    def train(self, epochs: int):
        def auc_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            pos = tf.boolean_mask(y_pred, y_true == 1)
            neg = tf.boolean_mask(y_pred, y_true == 0)
            diff = tf.expand_dims(neg, 1) - tf.expand_dims(pos, 0)
            return tf.reduce_mean(tf.nn.sigmoid(diff))

        self.model.compile(
            loss=auc_loss,
            optimizer="adam",
            metrics=[AUC(name="roc_auc", curve="ROC")]
            )
        self.model.fit(self.X_train, self.y_train, epochs=epochs)

    def infere(self):
        self.probas_train = self.model.predict(self.X_train)
        self.probas_test = self.model.predict(self.X_test)

    def equity_curve(self, threshold: float, plot, train=False):
        if train:
            equity_curve(
                self.probas_train,
                self.data["Close"][self.train_size + self.len_sequence:],
                threshold,
                plot=plot
            )
        else:
            equity_curve(
                self.probas_test,
                self.data["Close"][self.train_size + self.len_sequence:],
                threshold,
                plot=plot
            )

    def pr_roc(self, plot, train=False):
        if train:
            print_pr_roc(self.y_train, self.probas_train, plot=plot)
        else:
            print_pr_roc(self.y_test, self.probas_test, plot=plot)


class TreeBasedModel():
    def __init__(
        self,
        data: pd.DataFrame,
        model,
        interval: int,
        change: float,
        features: list
    ):
        self.model = model
        self.data = data
        self.features = features
        self.change = change
        self.interval = interval

    def enrich_data(self):
        self.data = add_macro(self.data)
        self.data = add_technicals(self.data)

    def add_target(self):
        self.data = target_building_ups(self.data, self.interval, self.change)

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
