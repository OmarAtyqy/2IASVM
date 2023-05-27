from src.SVM import *
from src.helper import *

# generate linearly sperated data
X, y = generate_linear_seperable_data(100)

# run SVM
w, b = SVM(X, y, C=1, tol=1e-3, max_passes=5)

# plot data
seperator = [[w, b]]
plot_data(X, y, seperators=seperator)