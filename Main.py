import pandas as pd
import numpy as np
import matplotlib as mpl
import sklearn as skl

from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class GroupProject:
    def dataPreprocessing(self):
    self.train = pd.read_csv('Adult_train.tab', sep = '\t')


train = train.dropna()
train = train.drop(0)
impFreq = SimpleImputer(missing_values='?', strategy='most_frequent')
train_imp = impFreq.fit_transform(train)
train = DataFrame(train_imp, columns = list(train.columns))