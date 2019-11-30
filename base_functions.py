import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from pathlib import Path


def gen_data(n_samples, std):
    x_blobs, y_blobs = make_blobs(n_samples=n_samples, n_features=2, cluster_std=std, centers=3)
    ds = pd.DataFrame({
        'feat_1': x_blobs[:, 0],
        'feat_2': x_blobs[:, 1],
        'target': y_blobs
    })

    ds.to_csv(f"data/dataset-{n_samples}-{std}.csv", index=False)


def train_model(neighbors, data_path, n_samples, std):
    ds = pd.read_csv(data_path)
    knn = KNeighborsClassifier(neighbors)
    x, y = ds.drop(columns=['target']), ds['target']

    knn.fit(x, y)

    with open(f"model/trained-{n_samples}-{std}-{neighbors}.pkl", 'wb') as f:
        pickle.dump(knn, f)


def predict(data_path, model_path):
    ds = pd.read_csv(data_path)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    name_parts = Path(model_path).name.split("-")

    predictions_name = f"predictions-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}.txt"
    path = Path.cwd() / 'predictions' / predictions_name

    predictions = model.predict(ds.drop(columns=['target']))
    np.savetxt(path, predictions)


"""

@click.command()
@click.option("--n_samples", type=int)
@click.option("--std", type=int)
def cli(n_samples, std):
    gen_data(n_samples, std)


if __name__ == '__main__':
    cli()

"""
