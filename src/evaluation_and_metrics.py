from sklearn.metrics import (precision_recall_curve,
                             auc,
                             roc_curve,
                             roc_auc_score
                             )
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def print_pr_roc(y_test, y_pred_prob, plot=False):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    if plot:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    return roc_auc


def equity_curve(
    y_pred_scores,
    close_prices,
    threshold=0.5,
    days_ahead=3,
    initial_capital=100,
    plot=False,
):
    close_prices = np.array(close_prices)
    signals = (np.array(y_pred_scores) >= threshold).astype(int)

    capital = initial_capital
    curve = [capital]

    t = 0
    while t < len(close_prices) - days_ahead:
        if signals[t] == 1:
            ret = 1 + (
                ((close_prices[t + days_ahead] - close_prices[t]) /
                 close_prices[t]) / 3
            )
            capital *= ret
            curve.append(capital)
            t += 1
        else:
            curve.append(capital)
            t += 1

    while len(curve) < len(close_prices):
        curve.append(capital)

    equity_series = pd.Series(curve, name="equity")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(equity_series)
        plt.title("Model guided portfolio")
        plt.xlabel("Days")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    return (equity_series.iloc[-1] - initial_capital) / len(equity_series)
