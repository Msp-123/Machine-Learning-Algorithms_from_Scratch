1.) Import necessary libraries:

import numpy as np
from collections import Counter
Here, the required libraries are imported. numpy is used for numerical operations, and Counter from the collections module will help with vote counting.

2.) Define the Euclidean distance function:

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance
This function calculates the Euclidean distance between two data points, x1 and x2. 
Euclidean distance is a measure of the straight-line distance between two points in a multidimensional space. It's computed here using the formula: √((x1 - x2)^2).

3.) class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

Here, a KNN class is defined to encapsulate the KNN algorithm's functionality.

* __init__(self, k): The constructor initializes the KNN instance with the number of neighbors to consider, denoted by k.

* fit(self, X, y): This method takes training data X (features) and corresponding labels y and stores them within the KNN instance. This information will be used for making predictions later.

* predict(self, X): Given an array of data points X, this method predicts the labels for each data point. It does this by iterating through the data points in X and calling the _predict method for each data point.
