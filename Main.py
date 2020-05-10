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
        train = pd.read_csv('Adult_train.tab', sep = '\t', na_values = "?") #przy imporcie zamienia ? na NaN
        pd.set_option("display.max_columns", None)
        train = train.drop([0,1], axis=0) #usuwanie wierszy 0 i 1 tak jak ustaliliśmy, w jednej linijce
        #print(train)
        print(train.isnull().sum()) #w niektórych kolumnach jest prawie 2000 NaN
        impFreq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        train_imp = impFreq.fit_transform(train)
        train = DataFrame(train_imp, columns = list(train.columns))
        print(train.isnull().sum()) #zamiana się udała, nie mamy teraz wartości NaN
        print(train.dtypes) #wszystkie dane są typu obiektowego, których macierz korelacji nie widzi.
        # Trzeba pozamieniać wartości nueryczne / logiczne.
        # Najwięcej zabawy będzie z kolumną y - chcemy mieć bool, a mamy wszędzie wartości niezerowe,
        # więc wszytko zostanie przekonwertowane na True.
        # Używam więc SimpleImputer by zamienić '<=50K' na 0
        impFreq = SimpleImputer(missing_values='<=50K', strategy='constant', fill_value = 0)
        train_imp = impFreq.fit_transform(train)
        train = DataFrame(train_imp, columns = list(train.columns))
        train['y'] = train['y'].astype('bool') # zmiana typu danych na bool. 0 na False, dane niezmienione na True
        train['age'] = train['age'].astype('float') #zmiana wieku na float. musi być tu, jak było przed SimpleImputer dla y to nie działało
        print(train.dtypes) #widać, że age i y zostały zmienione
        print(train.corr()) #macierz korelacji. Liczylem na większą korelację wieku z dochodem niż 23%. Trzeba zamienić więcej kolumn na typy numeryczne

d = GroupProject()
d.dataPreprocessing()