from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


if __name__ == '__main__':

    f = open("mlpLogs.txt", "a")
    
    sizeLayer = [50.,100.,200.]
    alphas = [0.01, 0.1]
    numLayers = [4,8,12]
    kf = StratifiedKFold(n_splits=10, shuffle=False)
    unSkewParam = True
    remColParam = 0
    

    df=loadData(quant=True,unSkew=unSkewParam,remCols=remColParam)
    X = df.drop(columns=['price'])
    Y = df['price']

    
    for nL in numLayers:
        for sL in sizeLayer:
            f.write(f"----------------------------------{nL} X {sL}-----------------------------\n")
            for a in alphas:
                f.write(f"----------------------------------ALPHA = {a}-----------------------------\n")
                totalTrainRSME = 0
                totalTestRSME = 0
                totalOOB = 0
                for train_index, test_index in kf.split(X, Y.to_numpy()):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                    X_train, X_test = scaleTrainTest(X_train, X_test)
                    layers = np.ones(nL)*sL
                    reg = MLPRegressor(hidden_layer_sizes=layers,
                                    alpha = a,
                                    solver = 'adam',
                                    activation = 'relu',
                                    learning_rate_init=0.001,
                                    batch_size=400,
                                    learning_rate='adaptive',
                                    momentum=0.9,
                                    early_stopping=True,
                                    validation_fraction=0.1).fit(X_train,y_train)
                    
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