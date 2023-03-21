import numpy as np
from kernels import *


# support vector machine algorithm for binary classification
def SVM(X, y, C=1, tol=1e-3, max_passes=10, kernel=linear_kernel):
    """
    Support vector machine algorithm
    Args:
        X: data points
        y: labels
        C: regularization parameter
        tol: tolerance
        max_passes: maximum number of passes
        kernel: kernel function to use
    Returns:
        w: weights
        b: bias
    """
    m, n = X.shape
    alphas = np.zeros(m)
    b = 0
    passes = 0
    
    while passes < max_passes:
        num_changed_alphas = 0
        
        for i in range(m):
            Ei = np.dot(alphas*y, kernel(X, X[i])) + b - y[i]
            
            if ((y[i]*Ei < -tol and alphas[i] < C) or (y[i]*Ei > tol and alphas[i] > 0)):
                j = np.random.choice(list(range(i)) + list(range(i+1, m)))
                Ej = np.dot(alphas*y, kernel(X, X[j])) + b - y[j]
                
                alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                
                if y[i] == y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                
                if L == H:
                    continue
                
                eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
                
                if eta >= 0:
                    continue
                
                alphas[j] -= (y[j] * (Ei - Ej)) / eta
                alphas[j] = max(L, min(H, alphas[j]))
                
                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue
                
                alphas[i] += y[i]*y[j]*(alpha_j_old - alphas[j])
                
                b1 = b - Ei - y[i]*(alphas[i]-alpha_i_old)*kernel(X[i], X[i]) - y[j]*(alphas[j]-alpha_j_old)*kernel(X[i], X[j])
                b2 = b - Ej - y[i]*(alphas[i]-alpha_i_old)*kernel(X[i], X[j]) - y[j]*(alphas[j]-alpha_j_old)*kernel(X[j], X[j])
                
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                num_changed_alphas += 1
                
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    
    W = np.dot(alphas * y, X)
    return W, b

# support vector machine algorithm for multi class classification using one vs one method
def SVM_One_vs_One(X, y, C=1, tol=1e-3, max_passes=5):
    """
    Support vector machine algorithm for multi class classification using one vs one method
    Args:
        X: data points
        y: labels
        C: regularization parameter
        tol: tolerance
        max_passes: maximum number of passes
    Returns:
        w: weights
        b: bias
    """
    n = len(X)
    classes = np.unique(y)
    num_classes = len(classes)
    alpha = []
    b = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            idx = np.where((y == classes[i]) | (y == classes[j]))[0]
            X_i = X[idx]
            y_i = y[idx]
            y_i[y_i == classes[i]] = 1
            y_i[y_i == classes[j]] = -1
            w, b_i = SVM(X_i, y_i, C, tol, max_passes)

            alpha.append(w)
            b.append(b_i)

    return alpha, b