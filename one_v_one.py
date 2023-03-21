from CODE.SVM import SVM_One_vs_One
from CODE.helper import *

# generate multi class data
X, y = generate_multi_class_data(100)

# run multi class svm
alpha, b = SVM_One_vs_One(X, y, C=1, tol=1e-3, max_passes=5)

seperators = []
for i, j in zip(alpha, b):
    seperators.append([i, j])

# plot data
plot_data(X, y, seperators=seperators)