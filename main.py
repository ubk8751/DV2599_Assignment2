# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.linear_model import _logistic as logMod
from sklearn.tree import DecisionTreeClassifier as treMod

# Dunno what these do yet
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# API functions for discretizising
from data import make_category, generate_bins, discretizise, training_validation_split

# Import 
unclean_data = pd.read_csv('spambase.csv')
# Discretizise using totally not stolen API
data = discretizise()

def kkN(data):
    pass

def trees(data):
    pass

def logReg(data):
    pass

def ten_fold_cross_val():
    pass

def friedman():
    pass

def significance(level:float = 0.05, vals:list = []):
    if vals is []:
        return

def Nemeyi():
    pass

def main():
    # Generate kNN Classifier
    neighbourhood = kNN(data)

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
