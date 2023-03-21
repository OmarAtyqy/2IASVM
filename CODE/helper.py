import numpy as np
import matplotlib.pyplot as plt


# generate linearly sperated data
def generate_linear_seperable_data(n):
    """
    Generate linearly seperable 2D data labeled as either 1 or -1
    Args:
        n: number of data points
    Returns:
        X: data points
        y: labels
    """
    X = []
    y = []
    for _ in range(n):
        x1 = np.random.uniform(1, 4)
        x2 = np.random.uniform(-30, -20)
        X.append([x1, x2])
        y.append(1)

        x1 = np.random.uniform(8, 11)
        x2 = np.random.uniform(0, 10)
        X.append([x1, x2])
        y.append(-1)
    
    return np.array(X), np.array(y)

# generate non linearly sperated data
def generate_non_linear_seperable_data(n):
    """
    Generate non linearly seperable 2D data labeled as either 1 or -1
    Args:
        n: number of data points
    Returns:
        X: data points
        y: labels
    """
    one_x = []
    minus_one_x = []
    one_y = []
    minus_one_y = []
    for _ in range(n):
        x1 = np.random.uniform(1, 4)
        x2 = np.random.uniform(-30, -20)
        one_x.append([x1, x2])
        one_y.append(1)

        x1 = np.random.uniform(8, 11)
        x2 = np.random.uniform(0, 10)
        minus_one_x.append([x1, x2])
        minus_one_y.append(-1)
    
    # add noise
    n_noise = int(n * 0.1)
    for i in range(n_noise):
        one_y[i] = -1
        minus_one_y[i] = 1
    
    return np.array(one_x + minus_one_x), np.array(one_y + minus_one_y)

# generate multi class data
def generate_multi_class_data(n):
    """
    Generate multi class 2D data labeled as either 1, 2, or 3
    Args:
        n: number of data points
    Returns:
        X: data points
        y: labels
    """
    X = []
    y = []
    for _ in range(n):
        x1 = np.random.uniform(1, 4)
        x2 = np.random.uniform(-30, -20)
        X.append([x1, x2])
        y.append(1)

        x1 = np.random.uniform(8, 11)
        x2 = np.random.uniform(0, 10)
        X.append([x1, x2])
        y.append(2)

        x1 = np.random.uniform(5, 9)
        x2 = np.random.uniform(-10, -20)
        X.append([x1, x2])
        y.append(3)
    
    return np.array(X), np.array(y)


# plot data
def plot_data(X, y, seperators = None):
    """
    Plot 2D data
    Args:
        X: data points
        y: labels
        seperator: array where seperator[0] is the slope and seperator[1] is the y-intercept
    """
    for index, point in enumerate(X):
        if y[index] == 1:
            plt.scatter(point[0], point[1], color='blue')
        elif y[index] == -1:
            plt.scatter(point[0], point[1], color='red')
        elif y[index] == 2:
            plt.scatter(point[0], point[1], color='green')
        elif y[index] == 3:
            plt.scatter(point[0], point[1], color='yellow')
    
    if seperators != None:
        for separator in seperators:
            w1, w2 = separator[0]
            w0 = separator[1]

            slope = -w1 / w2
            intercept = -w0 / w2

            x = np.linspace(0, 12, 100)
            y = slope * x + intercept
            plt.plot(x, y, color='black') 

    plt.show()