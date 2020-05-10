import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl

from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

    def optimalKforkNN(self, X_train, y_train):
        neighbours = np.array([k for k in range(9, 21, 2)])
        cv_scores = []
        for k in neighbours:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())
        opt_k = neighbours[cv_scores.index(max(cv_scores))]
        plt.scatter(neighbours, cv_scores)
        plt.show()
        print("Optymalna wartość k:", opt_k)
        return opt_k

    def optimalMinSamplesSplit(self, X_train, y_train):
        min_samples_split = np.array([k for k in range(150,500,10)])
        cv_scores = []
        for k in min_samples_split:
            dt = DecisionTreeClassifier(min_samples_split = k)
            scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())
        opt_split = min_samples_split[cv_scores.index(max(cv_scores))]
        plt.scatter(min_samples_split, cv_scores)
        plt.show()
        print("Optymalna wartość split:", opt_split)
        return opt_split

    def getClassificationScore(self, clf_name ,y_test, y_pred):
        print("Nazwa klasyfikatora: " + clf_name)
        print("Confusion matrix", confusion_matrix(y_test, y_pred))
        print("accuracy_score", accuracy_score(y_test, y_pred))
        print("recall_score", recall_score(y_test, y_pred))
        # specyficzność ACC - recall
        print("specyficzność", accuracy_score(y_test, y_pred)-recall_score(y_test, y_pred))
        print("roc_auc_score", roc_auc_score(y_test, y_pred))
        print("f1_score", f1_score(y_test, y_pred))
        print("precision_score", precision_score(y_test, y_pred))

    def ensableClassifier(self, clfs, X_train, X_test, y_train):
        y_preds = []
        # trenowanie i testowanie wszystkich klasyfikatorów z listy clfs
        for clf in clfs:
            clf.fit(X_train, y_train)
            y_preds.append(clf.predict(X_test))
        # głosowanie większościowe
        y_result = y_preds[0]
        clf_index = 1
        while(clf_index < len(y_preds)):
            index = 0
            while(index < len(y_result)):
                y_result[index] = y_result[index] + y_preds[clf_index][index]
                index += 1
            clf_index += 1
        # uśrednianie i zaokrąglanie
        for index, y in enumerate(y_result):
            y_result[index] = round(y_result[index]/len(clfs))
        return y_result
    def plotClassificationResult(self, column1, x_label, column2, y_label, y_pred):
        plt.scatter(column1, column2, c=y_pred)
        plt.title("Klasyfikacja próbek")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


d = GroupProject()
d.dataPreprocessing()
X_train, X_test, y_train, y_test = d.splitDatasetIntoTrainAndTest() #przechwycywanie wartości funkcji
# #Model SVM
# y_pred_svm_lin_train = d.trainAndTestClassifier(SVC(kernel='linear'), X_train,X_train,y_train)
# y_pred_svm_lin_test = d.trainAndTestClassifier(SVC(kernel='linear'), X_train,X_test,y_train)
# y_pred_svm_rbf_train = d.trainAndTestClassifier(SVC(kernel='rbf', gamma='auto'), X_train,X_train,y_train)
# y_pred_svm_rbf_test = d.trainAndTestClassifier(SVC(kernel='rbf', gamma='auto'), X_train,X_test,y_train)
#
# #kNN
opt_k = d.optimalKforkNN(X_train, y_train)


# y_pred_knn_opt_train = d.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=opt_k), X_train,X_train,y_train)
# y_pred_knn_opt_test = d.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=opt_k), X_train,X_test,y_train)
# d.getClassificationScore("kNN_opt", y_train, y_pred_knn_opt_train)
# d.getClassificationScore("kNN_opt", y_test, y_pred_knn_opt_test)

# #Decision Tree

opt_split = d.optimalMinSamplesSplit(X_train,y_train)
# y_pred_tree_train = d.trainAndTestClassifier(DecisionTreeClassifier(min_samples_split = opt_split), X_train,X_train,y_train)
# y_pred_tree_test = d.trainAndTestClassifier(DecisionTreeClassifier(min_samples_split = opt_split), X_train,X_test,y_train)
# d.getClassificationScore("DecisionTreeTraining", y_train, y_pred_tree_train)
# d.getClassificationScore("DecisionTreeTesting", y_test, y_pred_tree_test)

#GaussianNB
# y_pred_gnb_train = d.trainAndTestClassifier(GaussianNB(), X_train,X_train,y_train)
# y_pred_gnb_test = d.trainAndTestClassifier(GaussianNB(), X_train,X_test,y_train)
# d.getClassificationScore("GNB trenowanie", y_train, y_pred_gnb_train)
# d.getClassificationScore("GNB testowanie", y_test, y_pred_gnb_test)


#Random Forrest
# y_pred_randomForest_train = d.trainAndTestClassifier(RandomForestClassifier(), X_train,X_train,y_train)
# y_pred_randomForest_test = d.trainAndTestClassifier(RandomForestClassifier(), X_train, X_test,y_train)
# d.getClassificationScore("RF", y_train, y_pred_randomForest_train)
# d.getClassificationScore("RF", y_test, y_pred_randomForest_test)

# klasyfikacja zespołowa
y_pred_ensable_train = d.ensableClassifier(
    [KNeighborsClassifier(n_neighbors=opt_k),
     DecisionTreeClassifier(min_samples_split = opt_split),
     RandomForestClassifier()], X_train, X_train, y_train)

y_pred_ensable_test = d.ensableClassifier(
    [KNeighborsClassifier(n_neighbors=opt_k),
     DecisionTreeClassifier(min_samples_split = opt_split),
     RandomForestClassifier()], X_train, X_test, y_train)

d.getClassificationScore("Uczenie zespołowe trenowanie", y_train, y_pred_ensable_train)
d.getClassificationScore("Uczenie zespołowe testowanie", y_test, y_pred_ensable_test)
# d.plotClassificationResult(X_test['age'],'age', X_test['sex'], 'sex', y_pred_ensable_test)