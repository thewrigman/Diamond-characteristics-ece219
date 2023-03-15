from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    f = open("ridgeLogs.txt", "a")
    funcNames = ["Nothing","Logrithm"]
    funcs = [nothing, np.log]
    scaleornot = [nothingtwo, scaleTrainTest]
    scaleNames = ["Not Scaled", "Scaled"]
    remColsGrid = [0]
    kf = StratifiedKFold(n_splits=10, shuffle=False)

    alphaGrid = [0.0001, 0.001,0.01, 0.1, 1, 10]

    for remColParam in remColsGrid:
        df=loadData(quant=True,unSkew=True,remCols=remColParam)
        for func in funcs: 
            X = df.drop(columns=['price'])
            Y = df['price']
            f.write(f"Using {funcNames[funcs.index(func)]}")
            f.write("---------------------------------------------------------\n")
            f.write(f"Num Columns Removed: {remColParam}\n")
            for alphaVal in alphaGrid:
                f.write(f"Alpha Val: {alphaVal}\n")
                for scaleParam in scaleornot:
                    f.write(f"{scaleNames[scaleornot.index(scaleParam)]}")
                    totalTrainRSME = 0
                    totalTestRSME = 0
                    for train_index, test_index in kf.split(X, Y.to_numpy()):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = Y.iloc[train_index].apply(func), Y.iloc[test_index].apply(func)
                        X_train, X_test = scaleParam(X_train, X_test)

                        reg = Lasso(alpha=alphaVal).fit(X_train,y_train)
                        trainPred = 0
                        testPred = 0
                        trainRSME = 0
                        testRSME = 0

                        if func == nothing:
                            trainPred = reg.predict(X_train)
                            testPred = reg.predict(X_test)
                            trainRSME = mean_squared_error(trainPred, y_train, squared=False)
                            testRSME = mean_squared_error(testPred, y_test, squared=False)

                        else:
                            trainPred = np.exp(reg.predict(X_train))
                            testPred = np.exp(reg.predict(X_test))
                            trainRSME = mean_squared_error(trainPred,np.exp(y_train), squared=False)
                            testRSME = mean_squared_error(testPred, np.exp(y_test), squared=False)

                        f.write(f"Split Training RSME: {trainRSME} Testing RSME {testRSME},\n")
                        totalTrainRSME += trainRSME
                        totalTestRSME += testRSME
                    f.write(f"Mean TrainRSME = {totalTrainRSME/10}\n")
                    f.write(f"Mean TestRSME = {totalTestRSME/10}\n")
            
    f.close()