from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':

    f = open("ranforLogs.txt", "a")
    max_feature = ["sqrt","log2",None]
    max_depths = [4, 20,100,200]
    max_trees = [10,50,100,200]
    kf = StratifiedKFold(n_splits=10, shuffle=False)
    unSkewParam = True
    remColParam = 0
    

    df=loadData(quant=True,unSkew=unSkewParam,remCols=remColParam)
    X = df.drop(columns=['price'])
    Y = df['price']


    f.write("---------------------------------------------------------\n")
    for maxFeature in max_feature:
        for maxDepth in max_depths:
            for maxTree in max_trees:
                totalTrainRSME = 0
                totalTestRSME = 0
                totalOOB = 0
                for train_index, test_index in kf.split(X, Y.to_numpy()):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                    X_train, X_test = scaleTrainTest(X_train, X_test)
                    reg = RandomForestRegressor(n_estimators=maxTree,
                                                max_features=maxFeature,
                                                max_depth = maxDepth,
                                                oob_score=True).fit(X_train,y_train)
                    trainPred = 0
                    testPred = 0
                    trainRSME = 0
                    testRSME = 0

                    trainPred = reg.predict(X_train)
                    testPred = reg.predict(X_test)
                    trainRSME = mean_squared_error(trainPred, y_train, squared=False)
                    testRSME = mean_squared_error(testPred, y_test, squared=False)
                    oob = reg.oob_score_
                    f.write(f"Split Training RSME: {trainRSME} Testing RSME {testRSME}, OOB Error: {oob}\n")
                    totalTrainRSME += trainRSME
                    totalTestRSME += testRSME
                    totalOOB += oob
                f.write(f"Mean TrainRSME = {totalTrainRSME/10}\n")
                f.write(f"Mean TestRSME = {totalTestRSME/10}\n")
                f.write(f"Mean OOB Error = {totalOOB/10}\n")
            
    f.close()