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

- __init__(self, k): The constructor initializes the KNN instance with the number of neighbors to consider, denoted by k.

- fit(self, X, y): This method takes training data X (features) and corresponding labels y and stores them within the KNN instance. 
This information will be used for making predictions later.

- predict(self, X): Given an array of data points X, this method predicts the labels for each data point. 
It does this by iterating through the data points in X and calling the _predict method for each data point.


4.) def _predict(self, x):
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]

The _predict method is used to predict the label for a single data point x. Here's what each step within _predict does:

- Calculate the Euclidean distance between the input data point x and all training data points stored in self.X_train. This produces a list of distances.

- Use np.argsort to obtain the indices of the k smallest distances, which correspond to the indices of the nearest neighbors.

- Retrieve the labels of the k nearest neighbors from self.y_train using the indices obtained in the previous step.

- Use Counter to count the occurrences of each label in the k_nearest_labels list. Counter returns a dictionary-like object with labels as keys and their counts as values.

- Use .most_common() on the Counter object to get a list of tuples sorted by counts in descending order. 
The majority-voted label (the label with the highest count) is the first element in this list.

- Return the predicted label based on the majority voting.     


In essence, the KNN class is designed to take in training data, store it, and then use it to predict labels for new data points based on the labels of the k nearest training neighbors 
using a majority voting mechanism.
