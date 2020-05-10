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
        train['workclass'] = train['workclass'].astype('string')
        train['fnlwgt'] = train['fnlwgt'].astype('float')
        train['education-num'] = train['education-num'].astype('float')
        train['marital-status'] = train['marital-status'].astype('string')
        train['occupation'] = train['occupation'].astype('string')
        train['relationship'] = train['relationship'].astype('string')
        train['race'] = train['race'].astype('string')
        train['sex'] = train['sex'].astype('string')
        train['capital-gain'] = train['capital-gain'].astype('float')
        train['capital-loss'] = train['capital-loss'].astype('float')
        train['hours-per-week'] = train['hours-per-week'].astype('float')
        train['native-country'] = train['native-country'].astype('string')
        train = train.drop(["education"], axis=1)

        workclass_mapper = {}
        for index, num_cat in enumerate(train.workclass.unique()):
            workclass_mapper[num_cat] = index
        train["workclass"] = train["workclass"].map(workclass_mapper)

        marital_status_mapper = {}
        for index, num_cat in enumerate(train['marital-status'].unique()):
            marital_status_mapper[num_cat] = index
        train["marital-status"] = train["marital-status"].map(marital_status_mapper)

        occupation_mapper = {}
        for index, num_cat in enumerate(train['occupation'].unique()):
            occupation_mapper[num_cat] = index
        train["occupation"] = train["occupation"].map(occupation_mapper)

        relationship_mapper = {}
        for index, num_cat in enumerate(train['relationship'].unique()):
            relationship_mapper[num_cat] = index
        train["relationship"] = train["relationship"].map(relationship_mapper)

        race_mapper = {}
        for index, num_cat in enumerate(train['race'].unique()):
            race_mapper[num_cat] = index
        train["race"] = train["race"].map(race_mapper)

        sex_mapper = {}
        for index, num_cat in enumerate(train['sex'].unique()):
            sex_mapper[num_cat] = index
        train["sex"] = train["sex"].map(sex_mapper)

        native_country_mapper = {}
        for index, num_cat in enumerate(train['native-country'].unique()):
            native_country_mapper[num_cat] = index
        train["native-country"] = train["native-country"].map(native_country_mapper)


        print(train.dtypes) #widać, że age i y zostały zmienione
        print(train.corr()) #macierz korelacji. Liczylem na większą korelację wieku z dochodem niż 23%. Trzeba zamienić więcej kolumn na typy numeryczne
        self.train = train


    def splitDatasetIntoTrainAndTest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.train.drop(['y'], axis=1), self.train['y'], test_size=0.2)
        print(X_test)
        print(y_test)
        return X_train, X_test, y_train, y_test

    def trainAndTestClassifier(self, clf, X_train, X_test, y_train):
        print(clf)
        # trenowanie
        clf.fit(X_train, y_train)
        # testowanie
        y_pred = clf.predict(X_test)
        return y_pred

    def getClassificationScore(self, clf_name ,y_test, y_pred):
        print("Nazwa klasyfikatora: " + clf_name)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))


d = GroupProject()
d.dataPreprocessing()
X_train, X_test, y_train, y_test = d.splitDatasetIntoTrainAndTest() #przechwycywanie wartości funkcji
y_pred_knn5_train = d.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_train,y_train)
y_pred_knn5_test = d.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_test,y_train)
d.getClassificationScore("kNN-5 trenowanie", y_train, y_pred_knn5_train)
d.getClassificationScore("kNN-5 testowanie", y_test, y_pred_knn5_test)