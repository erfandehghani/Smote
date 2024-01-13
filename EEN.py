from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import pandas as pd
class ENN :
  def __init__(self,sample,sampling_strategy,n_neighbors) :
    self.sampling_strategy = sampling_strategy
    self.n_neighbors = n_neighbors
    self.sample = sample

  def class_divider(self):
        cancer_data_frame = pd.DataFrame(self.sample.data, columns=self.sample.feature_names)
        cancer_data_frame['Label'] = self.sample.target
        positive_samples = cancer_data_frame[cancer_data_frame['Label'] == 1].copy()
        positive_samples = positive_samples.reset_index()
        negative_samples = cancer_data_frame[cancer_data_frame['Label'] == 0].copy()
        negative_samples = negative_samples.reset_index()

        if len(positive_samples) > len(negative_samples) :
          majority_class = positive_samples
          minority_class = negative_samples

        else :
          majority_class = negative_samples
          minority_class = positive_samples


        return minority_class , majority_class

  def set_enn_class(self):
        minority_class , majority_class = self.class_divider()
        enn_class = minority_class if self.sampling_strategy == "minority" else majority_class
        return enn_class,self.sampling_strategy

  def EditedNearestNeigbor(self):
      minority_class , majority_class = self.class_divider()
      enn_class,type_of_enn_class =self.set_enn_class()
      knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
      knn.fit(enn_class.iloc[:, :-1], enn_class['Label'])


      edited_samples = []
      for idx, row in enn_class.iterrows():
            neighbor_indices = knn.kneighbors([row.drop('Label')], n_neighbors=self.n_neighbors, return_distance=False)[0]
            neighbor_labels = enn_class.iloc[neighbor_indices]['Label'].values
            if (row['Label'] == 1 and sum(neighbor_labels == 0) >= 3) or (row['Label'] == 0 and sum(neighbor_labels == 1) >= 3):
                edited_samples.append(idx)

      edited_data = enn_class.loc[edited_samples]
      secondery_class = majority_class if type_of_enn_class == "minority" else minority_class

      combinde_sample = pd.concat([edited_data,secondery_class], ignore_index=True)
      return combinde_sample





breast_cancer = load_breast_cancer()
print("len of sample befor enn : ",len(breast_cancer.data))


X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=42)

knn_before_sampling = KNeighborsClassifier(n_neighbors=5)
knn_before_sampling.fit(X_train, y_train)

y_pred_before_sampling = knn_before_sampling.predict(X_test)

accuracy_before_sampling = accuracy_score(y_test, y_pred_before_sampling)
print(f"Accuracy before under-sampling: {accuracy_before_sampling:.4f}")

enn = ENN(sample=breast_cancer, sampling_strategy="majority", n_neighbors=5)
under_sampled_data = enn.EditedNearestNeigbor()
print("len of sample after enn : ",len(under_sampled_data))

X_under_sampled = under_sampled_data.drop('Label', axis=1)
y_under_sampled = under_sampled_data['Label']

X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_under_sampled, y_under_sampled, test_size=0.2, random_state=42)

knn_after_sampling = KNeighborsClassifier(n_neighbors=5)
knn_after_sampling.fit(X_train_us, y_train_us)

y_pred_after_sampling = knn_after_sampling.predict(X_test_us)

accuracy_after_sampling = accuracy_score(y_test_us, y_pred_after_sampling)
print(f"Accuracy after under-sampling: {accuracy_after_sampling:.4f}")