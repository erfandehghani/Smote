import array

import pandas
import random
import numpy
from sklearn.neighbors import NearestNeighbors


class Smote:
    def __init__(self, samples, over_sample_percentage, n_nearest_neighbors):

        self.samples = pandas.DataFrame(samples).to_numpy()

        self.overSampleAmount = over_sample_percentage
        self.nNearestNeighbors = n_nearest_neighbors + 1

        self.numberOfAttributes = len(self.samples[0])

        self.t = len(self.samples)

        self.createdSamples = 0

        self.samplesNearestNeighbors = self.set_nearest_neighbors()

        self.randomize_samples()

        self.synthetics = self.create_synthetic_samples()

    def randomize_samples(self):
        if self.overSampleAmount < 100:
            self.t = int(self.overSampleAmount / 100 * self.t)
            numpy.random.shuffle(self.samples)
            self.overSampleAmount = 1
        else:
            self.overSampleAmount = self.overSampleAmount / 100

    def get_nearest_neighbors(self):
        return self.samplesNearestNeighbors[:, 1:self.nNearestNeighbors]

    def set_nearest_neighbors(self):
        nearest_neighbor = NearestNeighbors(n_neighbors=self.nNearestNeighbors)
        nearest_neighbor.fit(self.samples)
        return nearest_neighbor.kneighbors(self.samples, return_distance=False)

    def create_synthetic_samples(self):

        synthetic_index = 0
        synthetic_samples = numpy.zeros((int(len(self.samples) * self.overSampleAmount), int(self.numberOfAttributes)))

        for i in range(0, self.t):

            over_sampling_amount = self.overSampleAmount

            while over_sampling_amount != 0:
                random_neighbor = numpy.random.choice(self.nNearestNeighbors)

                for attribute in range(self.numberOfAttributes):
                    selected_nearest_neighbor = self.samplesNearestNeighbors[i][random_neighbor]
                    gap = self.samples[selected_nearest_neighbor][attribute] - self.samples[i][attribute]
                    difference = numpy.random.uniform(0, 1)
                    synthetic_samples[synthetic_index][attribute] = self.samples[i][attribute] + gap * difference

                synthetic_index = synthetic_index + 1
                self.createdSamples += 1
                over_sampling_amount = over_sampling_amount - 1

        return synthetic_samples

    def get_created_sample_count(self):
        return self.createdSamples


    def get_synthetic_samples(self):
        return self.synthetics
