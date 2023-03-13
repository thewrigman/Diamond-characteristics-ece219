import numpy as np
import sys
import pandas as pd

################################################################################
sys.path.append("../project_data")
def loadData(filePath="../project_data/diamonds.csv"):
    df = pd.read_csv(filePath)
    return df

