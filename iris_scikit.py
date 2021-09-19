import csv
import math
from collections import Counter
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



filename = "datasets/iris.data"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []
types = []

def classification(name):

    if name == "Iris-setosa":
        return 0
    elif name == "Iris-versicolor":
        return 1
    else:
        return 2

def correlate_freqs(v1,v2):

    num_correct = 0

    for i in range(len(v1)):
        
        if (v1[i] == classification(v2[i])):
            num_correct += 1

    return num_correct

def normalization(array):
    # maximum = max(array)
    # minimum = min(array)
    normalized_list = []

    #print(total_set)

    maximum = np.maximum.reduce(total_set)
    minimum = np.minimum.reduce(total_set)

    for i in range(len(array)):
        e = array[i]

        normalized_list.append((e - minimum[i]) / (maximum[i] - minimum[i]))

    return normalized_list

#https://www.geeksforgeeks.org/working-csv-files-python/
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
      
    # extracting field names through first row
    fields = next(csvreader)

    shuffled = []

    # extracting each data row one by one
    for row in csvreader:

        shuffled.append(row)

    random.seed(4)
    random.shuffle(shuffled)

    for row in shuffled:

        # print(row)

        floated_row = [float(row[i]) for i in range(len(row)-1)]
        total_set.append(floated_row)
        # print(row)
        # print(row[-1])
        types.append(row[-1])

        # print(floated_row)

    for row in range(len(total_set)):
        normalized_set.append(normalization(total_set[row]))
    # print(normalized_set)


#W/O Weights

X_train, X_test, y_train, y_test = train_test_split(normalized_set, types, test_size=0.5, random_state=1)

# Create KNN classifier
knn_unweighted = KNeighborsClassifier(n_neighbors = 22, weights='uniform')
# Fit the classifier to the data
knn_unweighted.fit(X_train,y_train)

print(knn_unweighted.score(X_test, y_test))


X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(normalized_set, types, test_size=0.5, random_state=1)

knn_weighted = KNeighborsClassifier(n_neighbors = 22, weights='distance')
# Fit the classifier to the data
knn_weighted.fit(X_train_w,y_train_w)

print(knn_weighted.score(X_test_w, y_test_w))











