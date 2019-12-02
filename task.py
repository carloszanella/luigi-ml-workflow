import luigi
from base_functions import gen_data, train_model, predict, make_plots

# 1. Generate Data
class GenerateData(luigi.Task):

    n_samples = luigi.IntParameter()
    std = luigi.FloatParameter()

    # First Task, doesn't require anything

    def output(self):
        # The generated data
        path = f"./data/dataset-{self.n_samples}-{self.std}.csv"
        return luigi.LocalTarget(path)

    def run(self):
        # Generates the random data
        gen_data(self.n_samples, self.std)


# 2. Train Model
class TrainModel(luigi.Task):

    n_samples = luigi.IntParameter()
    std = luigi.FloatParameter()
    neighbors = luigi.IntParameter(default=5)

    def requires(self):
        return GenerateData(n_samples=self.n_samples, std=self.std)

    def output(self):
        path = f"./model/trained-{self.n_samples}-{self.std}-{self.neighbors}.pkl"
        return luigi.LocalTarget(path)

    def run(self):
        # print('\n' + self.input().path + '\n')
        train_model(self.neighbors, self.input().path, self.n_samples, self.std)


# 3. Predict Values
class PredictValues(luigi.Task):

    n_samples = luigi.IntParameter()
    std = luigi.FloatParameter()
    neighbors = luigi.IntParameter(default=5)

    def requires(self):
        deps = [
            TrainModel(
                n_samples=self.n_samples, std=self.std, neighbors=self.neighbors
            ),
            GenerateData(n_samples=self.n_samples, std=self.std),
        ]
        return deps

    def output(self):
        path = f"./predictions/predictions-{self.n_samples}-{self.std}-{self.neighbors}.pkl.txt"
        return luigi.LocalTarget(path)

    def run(self):
        predict(self.input()[1].path, self.input()[0].path)


# 4. Plot results
class PlotResults(luigi.Task):

    n_samples = luigi.IntParameter()
    std = luigi.FloatParameter()
    neighbors = luigi.IntParameter(default=5)

    def requires(self):
        return PredictValues(
            n_samples=self.n_samples, std=self.std, neighbors=self.neighbors
        )

    def output(self):
        path = f"./results/results-{self.n_samples}-{self.std}-{self.neighbors}.png"
        return luigi.LocalTarget(path)

    def run(self):
        make_plots(self.n_samples, self.std, self.neighbors)
