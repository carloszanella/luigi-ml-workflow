import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def gen_data(n_samples, std):
    """
    Generates random data using sklearn's make_blobs based
    on the input parameters. Only 2 features are used, so the dataset
    shape is [n_samples x 3] (2 features + target column).

    The data is stored the path:
    f"data/dataset-{n_samples}-{std}.csv"

    Parameters
    ----------
    n_samples: int
        Number of rows for the output data
    std: float
        Standard deviation used in generating the data

    """

    x_blobs, y_blobs = make_blobs(
        n_samples=n_samples, n_features=2, cluster_std=std, centers=3
    )
    ds = pd.DataFrame(
        {"feat_1": x_blobs[:, 0], "feat_2": x_blobs[:, 1], "target": y_blobs}
    )

    ds.to_csv(f"data/dataset-{n_samples}-{std}.csv", index=False)


def train_model(neighbors, data_path, n_samples, std):
    """
    Trains a model using the data from `data_path`. A
    sklearn.neighbors.KNN(n_neighbors=neighbors) model is
    used for this task.

    Model is saved as pkl on:
    f"model/trained-{n_samples}-{std}-{neighbors}.pkl"

    Parameters
    ----------
    neighbors: int. Default=5
        Number of neighbors for KNN model
    data_path: string
        Path to the training data
    n_samples: int
        Number of rows in the data
    std: float
        Std deviation used to generate the data

    """
    ds = pd.read_csv(data_path)
    knn = KNeighborsClassifier(neighbors)
    x, y = ds.drop(columns=["target"]), ds["target"]

    knn.fit(x, y)

    with open(f"model/trained-{n_samples}-{std}-{neighbors}.pkl", "wb") as f:
        pickle.dump(knn, f)


def predict(data_path, model_path):
    """
    Makes predictions based on a model and a dataset. Both
    must match in the number of samples.

    Results are stored as txt in:
    f"predictions/predictions-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}.txt"

    Parameters
    ----------
    data_path: string
        Path to the dataset
    model_path: string
        Path to the model

    """
    ds = pd.read_csv(data_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    name_parts = Path(model_path).name.split("-")

    predictions_name = (
        f"predictions-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}.txt"
    )
    path = Path.cwd() / "predictions" / predictions_name

    predictions = model.predict(ds.drop(columns=["target"]))
    np.savetxt(path, predictions)


def make_plots(n_samples, std, neighbors=5):
    """
    Generates two results plots based on the true values and the predicted values.
    These plots are saved as png on:
    f"results/results-{n_samples}-{std}-{neighbors}.png"


    Parameters
    ----------
    n_samples: int
        Number of rows in the data
    std: float
        Std deviation used to generate the data
    neighbors: int. Default=5
        Number of neighbors for KNN model

    """

    p = Path()

    data_path = p / "data" / f"dataset-{n_samples}-{std}.csv"
    pred_path = p / "predictions" / f"predictions-{n_samples}-{std}-{neighbors}.pkl.txt"
    output_path = p / "results" / f"results-{n_samples}-{std}-{neighbors}.png"

    df = pd.read_csv(data_path)
    df["predictions"] = np.loadtxt(pred_path)
    df["correct"] = df["predictions"] == df["target"]

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 10))
    sns.scatterplot(data=df, x="feat_1", y="feat_2", hue="predictions", ax=axes[0])
    sns.scatterplot(data=df, x="feat_1", y="feat_2", hue="correct", ax=axes[1])
    plt.savefig(output_path)
    plt.close()
