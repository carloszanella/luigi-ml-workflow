import luigi
from base_functions import gen_data, train_model

# 1. Generate Data
class GenerateData(luigi.Task):

    n_samples = luigi.IntParameter()
    std = luigi.FloatParameter()

    # First Task, doesn't require anything
    # def requires(self):
    #     pass

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
        train_model(self.neighbors, self.input().path, self.n_samples, self.std)


# 3. Predict Values
class PredictValues(luigi.Task):

    n_samples = luigi.IntParameter()
    std = luigi.FloatParameter()
    neighbors = luigi.IntParameter(default=5)

    def requires(self):
        return TrainModel(n_samples=self.n_samples, std=self.std, neighbors=self.neighbors)

    def output(self):
        path = f"./model/trained-{self.n_samples}-{self.std}-{self.neighbors}.pkl"
        return luigi.LocalTarget(path)

    def run(self):
        pass


# 4. Evaluate Performance
class EvaluatePerformance(luigi.Task):
    def requires(self):
        pass

    def output(self):
        pass

    def run(self):
        pass
