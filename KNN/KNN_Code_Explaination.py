1.) Import necessary libraries:

import numpy as np
from collections import Counter

2.) Define the Euclidean distance function:

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance
This function calculates the Euclidean distance between two data points x1 and x2.

3.) 
