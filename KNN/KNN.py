import numpy as np
from collections import Counter

# Euclidean distance function calculates the distance between two points in a space.
# It's used here to measure the distance between data points.
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

# KNN (K-Nearest Neighbors) class is defined to perform classification using the KNN algorithm.
class KNN:
    def __init__(self, k):
        self.k = k  # The number of nearest neighbors to consider
        
    def fit(self, X, y):
        # Store the training data and their corresponding labels
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        # For each data point in X, predict its label using the KNN algorithm
        predictions = [self._predict(x) for x in X]
        return predictions
        
    def _predict(self, x):
        # Given a single data point 'x', predict its label
        
        # Compute the distance between the selected data point 'x' and all other data points in training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Find the indices of the 'k' closest data points
        k_indices = np.argsort(distances)[:self.k]
        
        # Retrieve the labels of the 'k' nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Perform majority voting among the 'k' nearest neighbor labels to predict the label for 'x'
        most_common = Counter(k_nearest_labels).most_common()
        
        # Return the predicted label based on majority voting
        return most_common[0][0]
