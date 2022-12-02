from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def ten_fold_cross_val(models, x, y):
    cross_val = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = []
    for model in models:
        m = model[1]
        score = cross_val_score(m, x, y, scoring="accuracy", cv=cross_val, n_jobs=-1)
        scores.append((model[0], list(score)))
    return scores