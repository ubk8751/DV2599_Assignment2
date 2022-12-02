from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score
from math import fsum

def ten_fold_cross_val(models, x, y):
    cross_val = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = []
    for model in models:
        m = model[1]
        score = cross_val_score(m, x, y, scoring="accuracy", cv=cross_val, n_jobs=-1)
        scores.append((model[0], list(score)))
    return scores

def friedman(alg, n_ranks):
    name = alg[0]
    scores = alg[1]
    histo, bin_edges = np.histogram(scores, bins=n_ranks)
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

def issignificant(data, level):
    sig_levels= {
                0.05: { 4: 6.50, 5: 6.40, 6: 7.00, 7: 7.143, 8: 6.25, 9: 6.222, 10: 6.20, 11: 6.545, 12: 6.50, 13: 6.615, 14: 6.143, 15: 6.40 },
                0.01: { 4: 8.00, 5: 8.40, 6: 9.00, 7: 8.857, 8: 9.00, 9: 9.556, 10: 9.60, 11: 9.455, 12: 9.50, 13: 9.385, 14: 9.143, 15: 8.933 }
            }
    totals = []
    vals = []
    for item in data:
        lst = []
        for rank in item["frank"]:
            lst.append(rank[1])
        vals.append(lst)  
    for i in vals:
        totals.append(fsum(i))
    total = 0
    for item in data:
        total += item["avg_rank"]
    avg_avg_rank = total/len(totals)
    sd = 0
    for item in data:    
        print(item["avg_rank"])
    for item in data:
        sd += (item["avg_rank"] - avg_avg_rank)**2
    #sd *= len(vals)
    print(sd)
    #val = (len(vals[0])/(len(vals[0]) * len(totals) * (len(totals) + 1))) * (totals[0]**2 + totals[1]**2 + totals[2]**2) - (len(vals[0]) * len(totals) * (len(totals) + 1))
    #print((len(vals[0]) * len(totals) * (len(totals) + 1)))
    
    #if val > sig_levels[level][len(vals[0])]:
    #    return sig_levels[level][len(data[0])]
    
    