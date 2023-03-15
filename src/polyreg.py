from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import os



if __name__ == '__main__':

    f = open("polyLogs.txt", "a")

    remColsGrid = [2,3, 4, 5, 6, 7]
    alphaGrid = [0.0001, 0.001,0.01, 0.1, 1, 10]

    kf = StratifiedKFold(n_splits=10, shuffle=False)

    for remColParam in remColsGrid:
        df=loadData(quant=True,unSkew=True,remCols=remColParam)
        X = df.drop(columns=['price'])
        Y = df['price']
        f.write("---------------------------------------------------------\n")
        f.write(f"Num Columns Removed: {remColParam}\n")



        poly = PolynomialFeatures(degree=9-remColParam, \
                                  interaction_only=False, \
                                  include_bias=True)
        poly.fit_transform(X)

        for alphaVal in alphaGrid:
            totalTrainRSME = 0
            totalTestRSME = 0

            for train_index, test_index in kf.split(X, Y):
            
                f.write(f"Alpha Val: {alphaVal}\n")

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                X_train, X_test = scaleTrainTest(X_train, X_test)
                
                reg = Ridge(alpha=alphaVal).fit(X_train,y_train)


                trainPred = reg.predict(X_train)
                testPred = reg.predict(X_test)

                trainRSME = mean_squared_error(trainPred, y_train, squared=False)
                explainedvar = explained_variance_score(trainPred, y_train)
                testRSME = mean_squared_error(testPred, y_test, squared=False)

                f.write(f"Split Training RSME: {trainRSME} Testing RSME {testRSME}, ev = {explainedvar}\n")
                totalTrainRSME += trainRSME
                totalTestRSME += testRSME

            f.write(f"Mean TrainRSME = {totalTrainRSME/10}\n")
            f.write(f"Mean TestRSME = {totalTestRSME/10}\n")
    
    f.close()
