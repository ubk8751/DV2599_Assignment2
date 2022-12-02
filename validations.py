from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score

def ten_fold_cross_val(models, x, y):
    cross_val = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = []
    for model in models:
        m = model[1]
        score = cross_val_score(m, x, y, scoring="accuracy", cv=cross_val, n_jobs=-1)
        scores.append((model[0], list(score)))
    return scores

def friedman(alg):
    name = alg[0]
    scores = alg[1]
    histo, bin_edges = np.histogram(scores, bins=3)
    third = (max(scores) - min(scores))/3
    frank = []
    for score in scores:
        if score <= (min(scores) + third):
            frank.append([score, 1])
        elif score <= (min(scores) + 2*third):
            frank.append([score, 2])
        elif score > (min(scores) + 2*third):
            frank.append([score, 3])
    total_score = 0
    for rank in frank:
        total_score += rank[1]
    avg = total_score/len(frank)
    fried_res = {"name": name, "frank": frank, "avg_rank": avg}
    return fried_res