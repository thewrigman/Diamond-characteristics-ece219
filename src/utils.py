import numpy as np
import sys
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    standardizedTrain = pd.DataFrame(scaler.fit_transform(train),columns=train.columns)
    standardizedTest = pd.DataFrame(scaler.transform(test),columns=test.columns)
    
    X_train= standardizedTrain.drop(columns=['price'])
    X_test = standardizedTest.drop(columns=['price'])
    y_train = train['price']
    y_test = test['price']

    return X_train, X_test, y_train, y_test




