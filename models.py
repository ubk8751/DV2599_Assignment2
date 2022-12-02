from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def kNeigbours(trx_data, try_data, tex_data):
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(trx_data, try_data)
    return knn

def trees(trx_data, try_data, tex_data):
    tree = DecisionTreeClassifier()
    tree.fit(trx_data, try_data)
    return tree

def logReg(trx_data, try_data, tex_data):
    longboi = LogisticRegression(max_iter=200)
    longboi.fit(trx_data, try_data)
    return longboi

def gen_models(trx_data, try_data, tex_data):
    # Generate kNN Classifier
    neighbourhood = kNeigbours(trx_data, try_data, tex_data)

    # Generate Decision Tree Classifier
    forest = trees(trx_data, try_data, tex_data)

    # Genrate Logarithmic Regression Classifier
    longboi = logReg(trx_data, try_data, tex_data)

    return ("K Nearest Neighbour Classifier", neighbourhood), ("Decision Tree Classifier", forest), ("Logarithmic Regression Classifier", longboi)
