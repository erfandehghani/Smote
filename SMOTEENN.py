from Smote import Smote
import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours


class SMOTEENN:
    VALID_STRATEGIES: set[str] = {"minority", "not minority", "not majority", "all"}

    def __init__(self, x, y, over_sample_percentage, n_nearest_neighbors=3,
                 sampling_strategy='not majority'):

        # Checking for valid sampling strategy options and raise an Error on Wrong input
        if sampling_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid sampling strategy: {sampling_strategy}. Valid options are: {self.VALID_STRATEGIES}")

        # Initializing properties
        self.sampling_strategy = sampling_strategy
        self.x = x
        self.y = y
        self.over_sample_percentage = over_sample_percentage
        self.n_nearest_neighbors = n_nearest_neighbors
        self.separated_x_classes = self.separate_x()
        self.separated_y_classes = self.separate_y()
        self.minority_samples, self.minority_targets = self.get_minority_class()
        self.majority_samples, self.majority_targets = self.get_majority_class()

    def get_x_and_y(self):
        return self.x, self.y

    def separate_x(self):
        """Separates the x samples into lists based on their corresponding class labels.

        Returns:
            dict: A dictionary where keys are the unique class labels and values are lists of x samples for each class.
        """
        unique_classes = np.unique(self.y)  # Get unique class labels

        class_data = {}
        for class_label in unique_classes:
            mask = self.y == class_label
            class_data[class_label] = self.x[mask]

        return class_data

    def separate_y(self):
        """Separates the y labels into lists based on their corresponding class labels.

        Returns:
            dict: A dictionary where keys are the unique class labels and values are lists of y labels for each class.
        """

        unique_classes = np.unique(self.y)  # Get unique class labels
        separated_y = {}

        for class_label in unique_classes:
            mask = self.y == class_label
            separated_y[class_label] = self.y[mask]

        return separated_y

    def get_corresponding_y(self, length_of_x):
        for i in range(len(self.separated_y_classes)):
            if len(self.separated_y_classes[i]) == length_of_x:
                return self.separated_y_classes[i]

    def get_majority_class(self):
        majority_class = self.separated_x_classes[0]
        for i in range(len(self.separated_x_classes)):
            if len(self.separated_x_classes[i]) > len(majority_class):
                majority_class = self.separated_x_classes[i]

        return majority_class, self.get_corresponding_y(len(majority_class))

    def get_minority_class(self):
        minority_class = self.separated_x_classes[0]
        for i in range(len(self.separated_x_classes)):
            if len(self.separated_x_classes[i]) < len(minority_class):
                minority_class = self.separated_x_classes[i]

        return minority_class, self.get_corresponding_y(len(minority_class))

    # def get_not_majority_classes(self):
    #     result_array = []
    #     for i in range(len(self.separated_x_classes)):
    #         if len(self.separated_x_classes[i]) != len(self.majority_samples):
    #             result_array.append(self.separated_x_classes[i])
    #     return result_array
    #
    # def get_not_minority_classes(self):
    #     result_array = []
    #     for i in range(len(self.separated_x_classes)):
    #         if len(self.separated_x_classes[i]) != len(self.minority_samples):
    #             result_array.append(self.separated_x_classes[i])
    #     return result_array

    def resample(self):
        self.smote_resample()
        self.enn_resample()

        return self.x, self.y

    # def resample_not_majority_class(self):
    #     not_majority_class = self.get_not_majority_classes()
    #
    # def resample_not_minority_class(self):
    #     not_minority_class = self.get_not_minority_classes()

    def enn_resample(self):
        if self.sampling_strategy == "minority":
            self.enn_majority_resample()

        # elif self.sampling_strategy == "not majority":
        #     return self.resample_not_majority_class()
        # elif self.sampling_strategy == "not minority":
        #     return self.resample_not_minority_Slass()
        # elif self.sampling_strategy == "all":
        #     return self.resample_all_class()

    def enn_majority_resample(self):
        enn = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=self.n_nearest_neighbors)
        self.x, self.y = enn.fit_resample(self.x, self.y)

    def smote_resample(self):
        synthetic_x, synthetic_y = self.smote_oversample()

        # Combining synthetic samples with the old ones
        self.x = pd.concat([pd.DataFrame(self.x), pd.DataFrame(synthetic_x)], ignore_index=True)
        self.y = pd.concat([pd.DataFrame(self.y), pd.DataFrame(synthetic_y)], ignore_index=True)

        return self.x, self.y

    # def resample_all_class(self):
    #     self.x = self.smote_oversample()

    def smote_oversample(self):
        smote = Smote(
            self.minority_samples,
            over_sample_percentage=self.over_sample_percentage,
            n_nearest_neighbors=self.n_nearest_neighbors)

        synthetic_x = smote.get_synthetic_samples()
        synthetic_y = pd.Series(self.minority_targets[0]).repeat(smote.get_created_sample_count())

        return synthetic_x, synthetic_y
