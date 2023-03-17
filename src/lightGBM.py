import numpy as np
import pandas as pd
import sklearn
from utils import *
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from lightgbm import LGBMRegressor
from skopt.space import Real, Categorical, Integer
import tempfile
import joblib
import json




# log-uniform: understand as search over p = exp(x) by varying x


#unskew True or False


#INSIDE GRID YOU HAVE HYPERPARAMETERS OF THE MODEL + whether or not you scale

def baeCV():
    scalers = Categorical((None, StandardScaler()))
    param_grid = [{'scaler': scalers, 'tree__num_leaves': Integer(8, 200), 'tree__max_depth': Integer(2,32), 
                   'tree__learning_rate': Real(.05, .5, prior='uniform'), 'tree__n_estimators': Integer(100, 200), 
                   'tree__min_split_gain': Real(0, .3)}]
    remColGrid = [0,2,4]
    steps = [('scaler', None), ('tree', LGBMRegressor(n_jobs=-1, random_state=42))]



    result_dict = {}
    for remColParam in remColGrid:
        df = loadData(quant=True,unSkew=True, remCols = remColParam)
        X = (df.drop(columns=['price'])).to_numpy()
        Y = (df['price']).to_numpy()
        cache_dir = tempfile.mkdtemp(dir='/Users/ineshchakrabarti/Documents/Diamond-characteristics-ece219/')
        mem = joblib.Memory(location=cache_dir, verbose=0)
        pipe = Pipeline(steps=steps, memory=mem)
        opt = BayesSearchCV(pipe, param_grid, cv=10, scoring='neg_root_mean_squared_error')
        opt.fit(X, Y)
        result_dict[remColParam] = {'best_score': opt.best_score_, 'best_params': str(opt.best_params_)}
        print("val score: %s" % opt.best_score_)
        print("best params: %s" % str(opt.best_params_))


    with open('/Users/ineshchakrabarti/Documents/Diamond-characteristics-ece219/bayes_search_results.txt', "w") as outfile:
        outfile.write(json.dumps(result_dict, indent=4))

if __name__ == "__main__":
    baeCV()
    

    
# python3 lightGBM.py > Bayes.txt