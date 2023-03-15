from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import os

if __name__ == '__main__':

    f = open("linregLog.txt", "a")

    funcs = [nothing, np.log]
    remColsGrid = [0]
    kf = StratifiedKFold(n_splits=10, shuffle=False)


    for func in funcs:
        for remColParam in remColsGrid:
            df=loadData(quant=True,unSkew=True,remCols=remColParam)
            X = df.drop(columns=['price'])
            Y = df['price']

            
            totalTrainRSME = 0
            totalTestRSME = 0
            f.write("---------------------------------------------------------\n")
            f.write(f"Num Columns Removed: {remColParam}\n")
            for train_index, test_index in kf.split(X, Y.to_numpy()):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index].apply(func), Y.iloc[test_index].apply(func)
                X_train, X_test = scaleTrainTest(X_train, X_test)

                reg = LinearRegression().fit(X_train,y_train)

                fig,axs = plt.subplots(1)
                axs.set_title('log Residuals for Linear Regression')
                axs.set_ylim(bottom=-1,top=1)
                axs.set_xlim(left=5, right=11)

                visualizer = ResidualsPlot(reg, ax=axs, train_alpha=0.01, test_alpha=0.05)
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test,y_test)
                visualizer.show()
                plt.savefig('linresplot.png')

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

                print(f"Split Training RSME: {trainRSME} Testing RSME {testRSME},\n")
                totalTrainRSME += trainRSME
                totalTestRSME += testRSME
            f.write(f"Mean TrainRSME = {totalTrainRSME/10}\n")
            f.write(f"Mean TestRSME = {totalTestRSME/10}\n")
            
            f.close()
