from sklearn.datasets import make_classification
import numpy as np


class SMOTEENN:
    VALID_STRATEGIES: set[str] = {"minority", "not minority", "not majority", "all"}

    def __init__(self, x_train, y_train, over_sample_percentage, n_nearest_neighbors=3,
                 sampling_strategy='not majority'):

        # Checking for valid sampling strategy options and raise an Error on Wrong input
        if sampling_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid sampling strategy: {sampling_strategy}. Valid options are: {self.VALID_STRATEGIES}")

        # Initializing properties
        self.sampling_strategy = sampling_strategy
        self.x_train = x_train
        self.y_train = y_train
        self.over_sample_percentage = over_sample_percentage
        self.n_nearest_neighbors = n_nearest_neighbors
        self.separated_classes = self.separate_x_train()
        self.minority_class = self.get_minority_class()
        self.majority_class = self.get_majority_class()

    def separate_x_train(self):
        unique_classes = np.unique(self.y_train)  # Get unique class labels

        class_data = {}
        for class_label in unique_classes:
            mask = self.y_train == class_label
            class_data[class_label] = self.x_train[mask]

        return class_data

    def get_majority_class(self):
        majority_class = self.separated_classes[0]
        for i in range(len(self.separated_classes)):
            if len(self.separated_classes[i]) > len(majority_class):
                majority_class = self.separated_classes[i]

        return majority_class

    def get_minority_class(self):
        minority_class = self.separated_classes[0]
        for i in range(len(self.separated_classes)):
            if len(self.separated_classes[i]) < len(minority_class):
                minority_class = self.separated_classes[i]

        return minority_class

    def get_not_majority_classes(self):
        result_array = []
        for i in range(len(self.separated_classes)):
            if len(self.separated_classes[i]) != len(self.majority_class):
                result_array.append(self.separated_classes[i])
        return result_array

    def get_not_minority_classes(self):
        result_array = []
        for i in range(len(self.separated_classes)):
            if len(self.separated_classes[i]) != len(self.minority_class):
                result_array.append(self.separated_classes[i])
        return result_array

    def resample_x_train(self):
        if self.sampling_strategy == "not majority":
            return self.resample_not_majority_class()

        elif self.sampling_strategy == "not minority":
            return self.resample_not_minority_class()

        elif self.sampling_strategy == "minoriry":
            return self.resample_minority_class()

        elif self.sampling_strategy == "all":
            return self.resample_all_class()

    def resample_not_majority_class(self):
        not_majority_class = self.get_not_majority_classes()

    def resample_not_minority_class(self):
        not_minority_class = self.get_not_minority_classes()