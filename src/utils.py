import numpy as np
import sys
import pandas as pd
import string
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
################################################################################
sys.path.append("../project_data")

def loadData(filePath="../project_data/diamonds.csv", quant = False, \
             unSkew = False, remCols = 0):
    df = pd.read_csv(filePath)
    df.drop(columns=["Unnamed: 0"],inplace=True) 
    if quant:
        df, ppc = qualtoquan(df)
    mutualInformationRanking = ['depth','table','cut','color','clarity',\
                                'z','x','y','carat']
    df.drop(columns=mutualInformationRanking[0:remCols], inplace=True)
    print(mutualInformationRanking[0:remCols])
    if unSkew:
        df = deSkew(df)
    return df

def scaleTrainTest(X_train, X_test):
    scaler = StandardScaler()
    xtrain = pd.DataFrame(scaler.fit_transform(X_train), \
                          columns = X_train.columns)
    xtest = pd.DataFrame(scaler.transform(X_test), \
                          columns = X_train.columns)
    return X_train, X_test

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

    ppc = df.copy()
    ppcarat = ppc['price']/ppc['carat']
    ppc.insert(0,'ppc', ppcarat)
    return df, ppc

def deSkew(data):
    methods = ["Sqrt", "boxcox", "No Change", "log"]
    processedDF = data.copy()
    for x in data.columns:
        if (x == 'cut' or x == 'color' or x == 'clarity'):
            processedDF[x] = data[x]
            print(f"{x}: Categorical")
            continue
        if (np.abs(data[x].skew()) < 0.5):
            print(f"{x}: No Change")
            continue
        logN = np.log(data[x])
        sqrtN = np.sqrt(data[x])
        boxcoxN = pd.Series(stats.boxcox(data['carat'])[0])
        Ns = [sqrtN, boxcoxN, data['x'], logN ]
        Skews = [sqrtN.skew(), boxcoxN.skew(),data[x].skew(),logN.skew()]
        bestSkew = min(Skews)
        bestMethod = Skews.index(bestSkew)
        processedDF[x] = Ns[bestMethod]
        print(f"{x}: {bestSkew}: {methods[bestMethod]}")
        processedDF['price'] = data['price']
    return processedDF


def scaledTrainTestSplit(data, ts=0.1):
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
