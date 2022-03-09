import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
df = pd.DataFrame(data=data, columns=columns)

X = df.drop(columns='target', axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

print(X_train.shape)
print(X_test.shape)

print(df)
