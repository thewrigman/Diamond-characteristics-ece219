from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    f = open("polyLogs2.txt", "a")
    funcNames = ["Nothing","Logrithm"]
    funcs = [nothing, np.log]
    # funcs = [nothing]
    kf = StratifiedKFold(n_splits=10, shuffle=False)
    remColsGrid = [3, 4, 5, 6, 7]
    alphaGrid = [0.0001, 0.001,0.01, 0.1, 1, 10]
    unSkewGrid = [True]
    for unSkewParam in unSkewGrid:
        if unSkewParam:
            f.write(f"Deskewed\n")
        else:
            f.write(f"Not deskewed\n")
        for remColParam in remColsGrid:
            f.write(f"DEGREE: {9-remColParam}")
            poly = PolynomialFeatures(degree=9-remColParam, interaction_only=False,\
                                        include_bias=True)
            df=loadData(quant=True,unSkew=unSkewParam,remCols=remColParam)
            for func in funcs:
                X = df.drop(columns=['price'])
                Y = df['price']
                X = pd.DataFrame(poly.fit_transform(X))
                f.write(f"Using {funcNames[funcs.index(func)]}")
                f.write("---------------------------------------------------------\n")
                f.write(f"Num Columns Removed: {remColParam}\n")
                for alphaVal in alphaGrid:
                    f.write(f"Alpha Val: {alphaVal}\n")
                    totalTrainRSME = 0
                    totalTestRSME = 0
                    try:
                        for train_index, test_index in kf.split(X, Y.to_numpy()):
                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                            y_train, y_test = Y.iloc[train_index].apply(func), Y.iloc[test_index].apply(func)
                            X_train, X_test = scaleTrainTest(X_train, X_test)

                            reg = Ridge(alpha=alphaVal).fit(X_train,y_train)
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
                    except:
                        f.write("\nOVERFLOW\n")
                        continue
            
    f.close()