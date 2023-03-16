import numpy as np
import pandas as pd
import sklearn
from utils import *


df = loadData()

# log-uniform: understand as search over p = exp(x) by varying x



opt = BayesSearchCV(
    StandardScaler(),
    n_iter=32,
    cv=3
)

opt.fit(X_train, y_train)




def bayesSearch():
    data = loadData(quant=True,unSkew=True)