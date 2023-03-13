import numpy as np
import sys
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
################################################################################
sys.path.append("../project_data")


def loadData(filePath="../project_data/diamonds.csv"):
    df = pd.read_csv(filePath)
    df.drop(columns=["Unnamed: 0"],inplace=True) 
    return df


def qualtoquan(data):
    df = data.copy(deep=True)
    cutLabels = ["Fair","Good","Very Good", "Premium", "Ideal"]
    for i in range(len(cutLabels)):
        df['cut'].replace(cutLabels[i],i+1,inplace=True)

    clarityLabels = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
    for i in range(len(clarityLabels)):
        df['clarity'].replace(clarityLabels[i],i+1,inplace=True)

    colorLabels = (list(string.ascii_uppercase[3:10]))
    for i in range (len(colorLabels)):
        df['color'].replace(colorLabels[i],len(colorLabels)-i,inplace=True)
    return df

def scaledTrainTest(data, ts=0.1):
    train, test = train_test_split(data,test_size=ts)

    scaler = StandardScaler()
    standardizedTrain = pd.DataFrame(scaler.fit_transform(train), \
                                     columns=train.columns)
    standardizedTest = pd.DataFrame(scaler.transform(test), \
                                    columns=test.columns)
    
    X_train= standardizedTrain.drop(columns=['price'])
    X_test = standardizedTest.drop(columns=['price'])
    y_train = train['price']
    y_test = test['price']

    return X_train, X_test, y_train, y_test

def plotMutualInformation(X_train, y_train):
    f1 = SelectKBest(score_func=mutual_info_regression, k='all')
    f1.fit(X_train, y_train)
    f2 = SelectKBest(score_func=f_regression, k='all')
    f2.fit(X_train, y_train)

    fig, axs = plt.subplots(2)
    axs[0].bar(X_train.columns, f1.scores_)
    axs[0].set_title("Mutual Information")
    axs[1].bar(X_train.columns, f2.scores_)
    axs[1].set_title("F Scores")
    fig.tight_layout()


    print("Mutual Information")
    for i in range(len(f1.scores_)):
        print(f"{X_train.columns[i]}: {f1.scores_[i]}")
    print("___________________________________________")
    print("F Scores")
    for i in range(len(f2.scores_)):
        print(f"{X_train.columns[i]}: {f2.scores_[i]}")
    return f1,f2

    



