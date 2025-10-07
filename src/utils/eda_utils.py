import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_ts(dataset: list, column: str):
    dataset["datetime"] = pd.to_datetime(dataset["Time"])
    dataset.set_index("datetime", inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(dataset.index, dataset[column])
    plt.title(f"{column} price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def volatility_analysis(periods: list, variations: list, data: pd.DataFrame):
    index = len(data["Close"])
    counts = {}
    for p in periods:
        for v in variations:
            counts[(p, v)] = 0
            for i in range(index - p):
                var = (
                    (data["Close"].iloc[i + p] - data["Close"].iloc[i])
                    / data["Close"].iloc[i]
                ) * 100
                if v >= 0:
                    if var >= v:
                        counts[(p, v)] += 1
                else:
                    if var <= v:
                        counts[(p, v)] += 1
    matrix = []
    for v in variations:
        row = []
        for p in periods:
            row.append(counts[(p, v)])
        matrix.append(row)

    heatmap_df = pd.DataFrame(
        matrix, index=[f"{v}%" for v in variations], columns=[f"{p}d" for p in periods]
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_df, annot=True, fmt="d", cbar_kws={"label": "Number of Occurrences"}
    )

    plt.title("volatility_analysis heatmap")
    plt.xlabel("Periods")
    plt.ylabel("Variations (%)")
    plt.tight_layout()
    plt.show()
    return counts
