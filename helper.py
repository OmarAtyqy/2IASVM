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


# plot data
def plot_data(X, y, seperator = None):
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
        else:
            plt.scatter(point[0], point[1], color='red')
    
    if seperator != None:
        w1, w2 = seperator[0]
        w0 = seperator[1]

        slope = -w1 / w2
        intercept = -w0 / w2

        x = np.linspace(0, 12, 100)
        y = slope * x + intercept
        plt.plot(x, y, color='black')   

    plt.show()