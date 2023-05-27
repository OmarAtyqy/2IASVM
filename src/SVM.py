import numpy as np
from .kernels import *


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
def SVM_One_vs_One(X, y, C=1, tol=1e-3, max_passes=5, kernel=linear_kernel):
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
            w, b_i = SVM(X_i, y_i, C, tol, max_passes, kernel=kernel)

            alpha.append(w)
            b.append(b_i)

    return alpha, b

# support vector machine algorithm for multi class classification using one vs all method
def SVM_One_vs_All(X, y, C=1, tol=1e-3, max_passes=10, kernel=linear_kernel):
    """
    One-vs-All Support Vector Machine algorithm
    Args:
        X: data points
        y: labels (with values 0 to num_classes-1)
        C: regularization parameter
        tol: tolerance
        max_passes: maximum number of passes
        kernel: kernel function to use
    Returns:
        classifiers: list of trained SVM classifiers (weights and biases)
    """
    num_classes = len(np.unique(y))

    alpha = []
    biases = []

    for class_label in range(num_classes):
        # Convert the class labels to binary (1 for the current class, -1 for the rest)
        binary_labels = np.where(y == class_label, 1, -1)
        
        # Train the SVM classifier for the current class
        W, b = SVM(X, binary_labels, C, tol, max_passes, kernel)

        alpha.append(W)
        biases.append(b)

        print(W, b)

    return alpha, biases

# support vector machine algorithm for multi class classification using  Error-correcting output coding method
def SVM_EOC(X, y, C=1, tol=1e-3, max_passes=10, kernel=linear_kernel):
    """
    Error-correcting output coding Support Vector Machine algorithm
    Args:
        X: data points
        y: labels (with values 0 to num_classes-1)
        C: regularization parameter
        tol: tolerance
        max_passes: maximum number of passes
        kernel: kernel function to use
    Returns:
        classifiers: list of trained SVM classifiers (weights and biases)
    """
    num_classes = len(np.unique(y))
    num_features = X.shape[1]

    # Initialize the code matrix
    code_matrix = np.zeros((num_classes, num_classes - 1))

    # Generate the code matrix
    for i in range(num_classes):
        code_matrix[i] = np.array([1 if j == i else -1 for j in range(num_classes) if j != i])

    # Initialize the classifiers list
    alphas, biases = [], []

    # Train the SVM classifiers
    for i in range(num_classes - 1):
        for j in range(i + 1, num_classes):
            # Get the indices of the data points that belong to the current pair of classes
            indices = np.where(np.logical_or(y == i, y == j))[0]

            # Get the corresponding labels for the current pair of classes
            labels = y[indices]
            labels = np.where(labels == i, 1, -1)

            # Get the corresponding data points for the current pair of classes
            data = X[indices]

            # Train the SVM classifier for the current pair of classes
            W, b = SVM(data, labels, C, tol, max_passes, kernel)

            # Store the weights and biases of the trained SVM classifier
            alphas.append(W)
            biases.append(b)

    return alphas, biases

# support vector machine algorithm for multi class classification using Ordinal regression machine method
def SVM_ORM(X, y, C=1, tol=1e-3, max_passes=10, kernel=linear_kernel):
    """
    Ordinal regression machine Support Vector Machine algorithm
    Args:
        X: data points
        y: labels (with values 0 to num_classes-1)
        C: regularization parameter
        tol: tolerance
        max_passes: maximum number of passes
        kernel: kernel function to use
    Returns:
        classifiers: list of trained SVM classifiers (weights and biases)
    """
    num_classes = len(np.unique(y))
    num_features = X.shape[1]

    # Initialize the classifiers list
    alphas, biases = [], []

    # Train the SVM classifiers
    for i in range(num_classes - 1):
        # Get the indices of the data points that belong to the current class and the next class
        indices = np.where(y <= i + 1)[0]

        # Get the corresponding labels for the current class and the next class
        labels = np.where(y[indices] == i, 1, -1)

        # Get the corresponding data points for the current class and the next class
        data = X[indices]

        # Train the SVM classifier for the current class and the next class
        W, b = SVM(data, labels, C, tol, max_passes, kernel)

        # Store the weights and biases of the trained SVM classifier
        alphas.append(W)
        biases.append(b)

    return alphas, biases

# support vector machine algorithm for multi class classification using Crammer-Singer multiclass SVM method
def SVM_Crammer_Singer(X, y, C=1, tol=1e-3, max_passes=10, kernel=linear_kernel):
    """
    Crammer-Singer multiclass SVM
    Args:
        X: data points
        y: labels (with values 0 to num_classes-1)
        C: regularization parameter
        tol: tolerance
        max_passes: maximum number of passes
        kernel: kernel function to use
    Returns:
        W: weights
        b: bias
    """
    num_classes = len(np.unique(y))
    num_features = X.shape[1]

    # Initialize the weights and biases
    W = np.zeros((num_classes, num_features))
    b = np.zeros(num_classes)

    # Train the SVM classifier
    for i in range(num_classes):
        # Get the indices of the data points that belong to the current class
        indices = np.where(y == i)[0]

        # Get the corresponding labels for the current class
        labels = np.ones(len(indices))

        # Get the corresponding data points for the current class
        data = X[indices]

        # Train the SVM classifier for the current class
        for j in range(num_classes):
            if j != i:
                # Get the indices of the data points that belong to the other class
                other_indices = np.where(y == j)[0]

                # Get the corresponding labels for the other class
                other_labels = -1 * np.ones(len(other_indices))

                # Get the corresponding data points for the other class
                other_data = X[other_indices]

                # Combine the data points and labels for the current class and the other class
                combined_data = np.concatenate((data, other_data))
                combined_labels = np.concatenate((labels, other_labels))

                # Train the SVM classifier for the current class and the other class
                W[i], b[i] = SVM(combined_data, combined_labels, C, tol, max_passes, kernel)

    return W, b