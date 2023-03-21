import numpy as np


# helper function
def f(alpha, y, X, x, b):
    return np.sum(alpha * y * np.dot(X, x.T)) + b

# prediction function
def predict(alpha, y, X, x, b):
    if f(alpha, y, X, x, b) >= 0:
        return 1
    else:
        return -1

# support vector machine algorithm
def SVM(X, y, C=1, tol=1e-3, max_passes=5):
    """
    Support vector machine algorithm
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
    alpha = np.zeros(shape=(n, 1))
    b = 0
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(n):
            E_i = f(alpha, y, X, X[i], b) - y[i]
            if (y[i]*E_i < -tol and alpha[i] < C) or (y[i]*E_i > tol and alpha[i] > 0):
                j = np.random.randint(0, n)
                while j == i:
                    j = np.random.randint(0, n)
                E_j = f(alpha, y, X, X[j], b) - y[j]
                old_alpha_i = alpha[i]
                old_alpha_j = alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue

                eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])

                if eta >= 0:
                    continue
                    
                alpha[j] -= y[j] * (E_i - E_j) / eta
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L

                if abs(alpha[j] - old_alpha_j) < 1e-10:
                    continue
                    
                alpha[i] += y[i] * y[j] * (old_alpha_j - alpha[j])

                b1 = b - E_i - y[i] * (alpha[i] - old_alpha_i) * np.dot(X[i], X[i]) - y[j] * (alpha[j] - old_alpha_j) * np.dot(X[i], X[j])
                b2 = b - E_j - y[i] * (alpha[i] - old_alpha_i) * np.dot(X[i], X[j]) - y[j] * (alpha[j] - old_alpha_j) * np.dot(X[j], X[j])

                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                num_changed_alphas += 1
            
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    
    # transform alpha to w
    w = np.sum(alpha * y.reshape(n, 1) * X, axis=0)

    return w, b