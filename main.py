# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Team written APIs
from models import gen_models
from data import discretizise
from validations import ten_fold_cross_val, friedman, issignificant, nemeyi

debug = True

def perform_friedman(ten_fold_data):

    # Friedman Tests
    friedman_res = friedman(ten_fold_data)
    
    # Determine significance on 0.5-alpha level
    sig = issignificant(friedman_res, 0.05)

    # Conduct Nemeyi test
    if sig:
        print('Test is significant, performing Nemeyi test')
        nemeyi_res = nemeyi(friedman_res)
        for a, b, diff in nemeyi_res:
            print(f'\tAlgorithms {a} and {b} exceeds critical difference {diff}')

    if debug:

        dfv = np.array([a[1] for a in ten_fold_data]).transpose()
        dfv_mean = np.mean(dfv, axis=0)
        dfv_std = np.std(dfv, axis=0)
        dfv = np.concatenate((dfv, dfv_mean.reshape(1,-1)), axis=0)
        dfv = np.concatenate((dfv, dfv_std.reshape(1,-1)), axis=0)
        fold_range = [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "avg", "stdev"]]

        print('\n\n----12.4-like print', end='')
        print('-' * 44)
        for name, _ in [('Fold', 0)] + ten_fold_data:
            print(f'{name.rjust(15)}', end=' ')
        print()
        for i in range(12):
            if i == 10:
                print('-' * 64)
            print(f'{fold_range[i].rjust(15)}{dfv[i][0]:16.4f}{dfv[i][1]:16.4f}{dfv[i][2]:16.4f}')
        print('-' * 64)


        print('\n\n----12.8-like print', end='')
        print('-' * 44)
        for name, _ in [('Fold', 0)] + ten_fold_data:
            print(f'{name.rjust(15)}', end=' ')
        print()

        r = friedman_res.astype(np.int32)
        dfv[10] = np.mean(r, axis=0)
        for i in range(10):
            print(f'{fold_range[i].rjust(15)}{dfv[i][0]:13.4f}({r[i][0]:1d}){dfv[i][1]:13.4f}({r[i][1]:1d}){dfv[i][2]:13.4f}({r[i][2]:1d})')
        print('-' * 64)
        print(f'{fold_range[10].rjust(15)}{dfv[10][0]:16.1f}{dfv[10][1]:16.1f}{dfv[10][2]:16.1f}')
        print('-' * 64)

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
    ten_fold_accuracy = ten_fold_cross_val([neighbourhood, forest, longboi], tex_data, tey_data, scoring="accuracy")
    ten_fold_f1_score = ten_fold_cross_val([neighbourhood, forest, longboi], tex_data, tey_data, scoring="f1")
    ten_fold_fit_time = ten_fold_cross_val([neighbourhood, forest, longboi], tex_data, tey_data, scoring="f1", measure_time=True)

    print(f'Friedman on accuracy:')
    perform_friedman(ten_fold_accuracy)
    
    print(f'Friedman on f1:')
    perform_friedman(ten_fold_f1_score)
    
    print(f'Friedman on time:')
    perform_friedman(ten_fold_fit_time)

    print("Finished with no errors :)")


if __name__ == "__main__":
    main()