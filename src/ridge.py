from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import os



if __name__ == '__main__':

    f = open("ridgeLogs.txt", "a")

    remColsGrid = [0]
    alphaGrid = [0.0001, 0.001,0.01, 0.1, 1, 10]

    kf = StratifiedKFold(n_splits=10, shuffle=False)

    for remColParam in remColsGrid:
        df=loadData(quant=True,unSkew=True,remCols=remColParam)
        X = df.drop(columns=['price'])
        Y = df['price']
        

        f.write("---------------------------------------------------------\n")
        f.write(f"Num Columns Removed: {remColParam}\n")
        for alphaVal in alphaGrid:
            stotalTrainRSME = 0
            stotalTestRSME = 0
            totalTrainRSME = 0
            totalTestRSME = 0
            for train_index, test_index in kf.split(X, Y):
            
                f.write(f"Alpha Val: {alphaVal}\n")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                sX_train, sX_test = scaleTrainTest(X_train, X_test)

                reg = Ridge(alpha=alphaVal).fit(sX_train,y_train)

                strainPred = reg.predict(sX_train)
                stestPred = reg.predict(sX_test)

                trainPred = reg.predict(X_train)
                testPred = reg.predict(X_test)

                strainRSME = mean_squared_error(strainPred, y_train, squared=False)
                stestRSME = mean_squared_error(stestPred, y_test, squared=False)

                trainRSME = mean_squared_error(trainPred, y_train, squared=False)
                testRSME = mean_squared_error(testPred, y_test, squared=False)

                f.write(f"Standard Training RSME: {strainRSME} Testing RSME {stestRSME}\n")
                f.write(f"NOT Standard Training RSME: {trainRSME} Testing RSME {testRSME}\n")
                stotalTrainRSME += strainRSME
                stotalTestRSME += stestRSME
                totalTrainRSME += trainRSME
                totalTestRSME += testRSME

            f.write(f"Standard Mean TrainRSME = {stotalTrainRSME/10}\n")
            f.write(f"Standard Mean TestRSME = {stotalTestRSME/10}\n")

            f.write(f"NOT Standard Mean TrainRSME = {totalTrainRSME/10}\n")
            f.write(f"NOT Standard Mean TestRSME = {totalTestRSME/10}\n")
    
    f.close()
