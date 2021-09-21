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

def normalization(array):
    #Normalize the attributes in respect to eachother in each column
    normalized_list = []

    maximum = np.maximum.reduce(total_set)
    minimum = np.minimum.reduce(total_set)

    for i in range(len(array)):
        e = array[i]

        normalized_list.append((e - minimum[i]) / (maximum[i] - minimum[i]))

    return normalized_list

#___________________________IMPLEMENTATION BELOW___________________________

#read and splice the data

#https://www.geeksforgeeks.org/working-csv-files-python/
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
      
    next(csvreader)

    for row in csvreader:

        floated_row = []

        for i in range(len(row)-1):
            if(row[i] != '?'):
                floated_row.append(float(row[i]))
            else:
                floated_row.append(0)

        total_set.append(floated_row)

        benign_or_malignant.append(row[-1])

for row in range(len(total_set)):
    normalized_set.append(normalization(total_set[row]))


#___________________________RUNNING BELOW____________________________________

k = 15

#Implementation reinterpreted from -> towardsdatascience.com
#Reference -> https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

#____________w/o weights_______________

X_train, X_test, y_train, y_test = train_test_split(normalized_set, benign_or_malignant, test_size=0.5, random_state=1)

# Create KNN classifier
knn_unweighted = KNeighborsClassifier(n_neighbors = k, weights='uniform')
# Fit the classifier to the data
knn_unweighted.fit(X_train,y_train)

print("Unweighted Correct Rate -> " + str(knn_unweighted.score(X_test, y_test))) #Print unweighted correct rate

#____________w/weights_______________


X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(normalized_set, benign_or_malignant, test_size=0.5, random_state=1)

knn_weighted = KNeighborsClassifier(n_neighbors = k, weights='distance')

# Fit the classifier to the data
knn_weighted.fit(X_train_w,y_train_w)

print("Weighted Correct Rate -> " + str(knn_weighted.score(X_test_w, y_test_w))) #Print weighted correct rate






