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
    #Associate the class name to a number
    if name == "Iris-setosa":
        return 0
    elif name == "Iris-versicolor":
        return 1
    else:
        return 2

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
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    

    shuffled = []

    # extracting each data row one by one
    for row in csvreader:
        shuffled.append(row)

    random.seed(4) #seed for replication
    random.shuffle(shuffled)

    for row in shuffled:

        floated_row = [float(row[i]) for i in range(len(row)-1)]
        total_set.append(floated_row)
        types.append(classification(row[-1]))


    for row in range(len(total_set)):
        normalized_set.append(normalization(total_set[row]))


#___________________________RUNNING BELOW____________________________________


#Implementation reinterpreted from -> towardsdatascience.com
#Reference -> https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

k = 22

#____________w/o weights_______________

X_train, X_test, y_train, y_test = train_test_split(normalized_set, types, test_size=0.5, random_state=1)

# Create KNN classifier
knn_unweighted = KNeighborsClassifier(n_neighbors = k, weights='uniform')
# Fit the classifier to the data
knn_unweighted.fit(X_train,y_train)

print("Unweighted Correct Rate -> " + str(knn_unweighted.score(X_test, y_test))) #Print unweighted correct rate

#____________w/weights_______________


X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(normalized_set, types, test_size=0.5, random_state=1)

knn_weighted = KNeighborsClassifier(n_neighbors = k, weights='distance')

# Fit the classifier to the data
knn_weighted.fit(X_train_w,y_train_w)

print("Weighted Correct Rate -> " + str(knn_weighted.score(X_test_w, y_test_w))) #Print weighted correct rate












