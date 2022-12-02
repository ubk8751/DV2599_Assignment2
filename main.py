# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Team written APIs
from models import gen_models
from data import discretizise
from validations import ten_fold_cross_val, friedman

debug = False

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

    # Generate models
    neighbourhood, forest, longboi = gen_models(trx_data, try_data, tex_data)

    # Perform ten-fold cross-validation tests
    validationMatrix = ten_fold_cross_val([neighbourhood, forest, longboi], tex_data, tey_data)
    if debug:
        for test in validationMatrix:
            print(test[0], end=": ")
            for i in test[1]:
                print(str(i), end=", ")
            print()

    # Friedman Tests
    test_results = []
    for alg in validationMatrix:
        test_results.append(friedman(alg))
    if debug == True:
        for result in test_results:
            print(result)

    vals = []

    # Determine significance on 0.5-alpha level
    zerofivesignificance = significance(0.5, vals)

    # Conduct Nemeyi test
    NemeyiResult = Nemeyi()

    print("Finished with no errors :)")


if __name__ == "__main__":
    main()