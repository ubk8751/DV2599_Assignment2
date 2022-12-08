from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from math import fsum, sqrt

def ten_fold_cross_val(models, x, y, scoring : str="accuracy", measure_time:bool=False):
    cross_val = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = []
    for data, model in models:
        m = model

        s = cross_validate(m, x, y, scoring=scoring, cv=cross_val, n_jobs=-1)
        score = s['test_score']
        if measure_time:
            score = s['fit_time']
            
        scores.append((data, list(score)))
    return scores

def friedman(data):

    df = [d[1] for d in data]
    df = np.array(df).transpose()

    rank_count = 3;
    fold_count = 10;
    ranks = []
    for row in range(fold_count):
        for i in range(rank_count, 0, -1):
            min_val = np.min(df[row,:])
            idx = np.where(min_val == df[row,:])[0]
            df[row,idx] = i

    return df

def issignificant(ranks, level):
    sig_levels= {                       # Significance levels taken from table
        0.05: { 4: 6.50, 5: 6.40, 6: 7.00, 7: 7.143, 8: 6.25, 9: 6.222, 10: 6.20, 11: 6.545, 12: 6.50, 13: 6.615, 14: 6.143, 15: 6.40 },
        0.01: { 4: 8.00, 5: 8.40, 6: 9.00, 7: 8.857, 8: 9.00, 9: 9.556, 10: 9.60, 11: 9.455, 12: 9.50, 13: 9.385, 14: 9.143, 15: 8.933 }
    } 
    sum_ranks = np.sum(ranks, axis=0)

    N = 10
    k = 3
    
    M = 12.0 / (N * k * (k + 1))
    S = sum([a**2 for a in sum_ranks])
    Q = 3 * N * (k + 1)

    friedman_val = M * S - Q
    return friedman_val > sig_levels[level][N]

def nemeyi(ranks):
    qa = 2.343
    k = 3
    n = 10
    CD = qa * sqrt(k * (k + 1) / (6 * n))
    
    avg_ranks = np.mean(ranks, axis=0)
    ret = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = np.abs(avg_ranks[i] - avg_ranks[j])
            if diff >= CD:
                ret.append((i, j, diff))
    return ret