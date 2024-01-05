from Smote import Smote
import pandas
from sklearn.datasets import load_iris

iris = load_iris()

test = Smote(pandas.DataFrame(iris.data).iloc[0:10, :], 200, 3)


print(test.get_synthetic_samples())
