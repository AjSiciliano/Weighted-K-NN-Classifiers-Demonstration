import csv
import math
from collections import Counter
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



filename = "datasets/breast-cancer-wisconsin.data"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []

benign_or_malignant = []

def correlate_freqs(v1,v2):

    num_correct = 0


    for i in range(len(v1)):
        
        if (v1[i] == int(v2[i])):
            num_correct += 1

    return num_correct


def normalization(array):

    normalized_list = []

    maximum = np.maximum.reduce(total_set)
    minimum = np.minimum.reduce(total_set)

    for i in range(len(array)):
        e = array[i]

        normalized_list.append((e - minimum[i]) / (maximum[i] - minimum[i]))

    return normalized_list


with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
      
    # extracting field names through first row
    fields = next(csvreader)

    shuffled = []

    # extracting each data row one by one
    for row in csvreader:

        shuffled.append(row)

    #random.shuffle(shuffled)

    for row in shuffled:

        # print(row)

        floated_row = []

        for i in range(len(row)-1):

            if(i != 0):
                if(row[i] != '?'):
                    floated_row.append(float(row[i]))
                else:
                    floated_row.append(0)


        total_set.append(floated_row)

        # print(row)
        # print(row[-1])
        benign_or_malignant.append(row[-1])


for row in range(len(total_set)):
    #print(total_set[row])
    normalized_set.append(normalization(total_set[row]))
    # print(normalized_set[row])

#W/O Weights

X_train, X_test, y_train, y_test = train_test_split(normalized_set, benign_or_malignant, test_size=0.5, random_state=1)

# Create KNN classifier
knn_unweighted = KNeighborsClassifier(n_neighbors = 15, weights='uniform')
# Fit the classifier to the data
knn_unweighted.fit(X_train,y_train)

print(knn_unweighted.score(X_test, y_test))


X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(normalized_set, benign_or_malignant, test_size=0.5, random_state=1)

knn_weighted = KNeighborsClassifier(n_neighbors = 15, weights='distance')
# Fit the classifier to the data
knn_weighted.fit(X_train_w,y_train_w)

print(knn_weighted.score(X_test_w, y_test_w))











