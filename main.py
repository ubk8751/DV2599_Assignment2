# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.linear_model import _logistic as logMod
from sklearn.tree import DecisionTreeClassifier as treMod
from sklearn.model_selection import train_test_split

# API functions for discretizising
from data import discretizise, training_validation_split

def kNeigbours(trx_data, try_data, tex_data):
    knn = kNN(n_neighbors=8)
    knn.fit(trx_data, try_data)
    return knn

def trees(data):
    print("Trees")

def logReg(data):
    print("logReg")

def ten_fold_cross_val():
    print("Crossval")

def friedman():
    print("Friedman")

def significance(level:float = 0.05, vals:list = []):
    print("significance")
    if vals is []:
        return

def Nemeyi():
    print("nemeyi")

def main():
    # Import data
    unclean_data = pd.read_csv('spambase.csv')

    # Discretizise using totally not stolen API
    old_data, data = discretizise()

    # Split dataset into Training- and Validation Data
    y_data = old_data["is_spam"]
    x_data = old_data.iloc[:,:-1]
    trx_data, tex_data, try_data, tey_data = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    # Generate kNN Classifier
    neighbourhood = kNeigbours(trx_data, try_data, tex_data)

    # Generate Decision Tree Classifier
    forest = trees(data)

    # Genrate Logarithmic Regression Classifier
    logboi = logReg(data)

    # Perform ten-fold cross-validation tests
    validationMatrix = ten_fold_cross_val()

    # Friedman Tests
    kNNFriedman = friedman()
    treeFriedman = friedman()
    logFriedman = friedman()
    
    vals = []

    # Determine significance on 0.5-alpha level
    zerofivesignificance = significance(0.5, vals)

    # Conduct Nemeyi test
    NemeyiResult = Nemeyi()

    print("Finished with no errors :)")


if __name__ == "__main__":
    main()