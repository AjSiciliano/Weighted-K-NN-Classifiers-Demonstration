import csv
import math
from collections import Counter
import numpy as np
import pandas as pd
import random

filename = "datasets/breast-cancer-wisconsin.data"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []
benign_or_malignant = []

def correlate_freqs(v1,v2):

    num_correct = 0

    for i in range(len(v1)):
        
        if (v1[i] == v2[i]):
            num_correct += 1

    return num_correct


def normalization(array):

    normalized_list = []

    maximum = np.maximum.reduce(full_data)
    minimum = np.minimum.reduce(full_data)

    for i in range(len(array)):
        e = array[i]

        normalized_list.append((e - minimum[i]) / (maximum[i] - minimum[i]))

    return normalized_list


df = pd.read_csv('datasets/breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
#drop unnecessary columns (id and class) ...also test without drop
df.drop(['id'],1,inplace=True)
df.drop(['class'],1,inplace=True)

#df has random quotes. Convert all to float
full_data = df.astype(float).values.tolist()
        
for row in range(len(full_data)):
    #print(total_set[row])
    normalized_set.append(normalization(full_data[row]))
    # print(normalized_set[row])

def euclidian_distance(row1, row2):
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)

mid = len(normalized_set)//2

def unweighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance(x,element), benign_or_malignant[index]])
        index += 1

    total_distances.sort()

    distances_to_k = [total_distances[i] for i in range(k)]

    #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
    above_dict = {2:[],4:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:
        above_dict[int(distance[-1])].append(distance[0])

    new_list = {0,0}

    for i in above_dict: 

        new_list[(i)] = len(above_dict[i])


    return max(new_list, key=new_list.get)

list_of_ks = [77]

set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(unweighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

    set_of_predictions.append(predictions)

total = 0

for x in set_of_predictions:
    
    total += correlate_freqs(x, benign_or_malignant[:mid])

total = total / len(set_of_predictions)

print( "Average error rate w/o weights: " + str(1 - (total / len(benign_or_malignant[:mid]))) )
print( "Average error rate w/o weights: " + str(total)) 

def weighted_KNN_classification(test_set, element,k):
    total_distances = []
    
    index = 0
    
    for x in test_set:
        print(x)
        total_distances.append([euclidian_distance(x,element),  benign_or_malignant[mid + index]])
        index += 1

    total_distances.sort()

    distances_to_k = [total_distances[i] for i in range(k)]
    distances = [total_distances[i][0] for i in range(len(distances_to_k))]


    maximum = max(distances)
    minimum = min(distances)

    above_dict = {2:[],4:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:
        above_dict[int(distance[-1])].append(distance[0])

    new_list = [0,0]

    for i in above_dict: 
       
        for d in above_dict[i]:

            if maximum != minimum:
                new_list[i] += ((maximum - d) / (maximum - minimum))
            else:
                new_list[i] += 1


    return new_list.index(max(new_list))
    



mid = len(normalized_set)//2

set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(weighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

    set_of_predictions.append(predictions)

total = 0

for x in set_of_predictions:
    # print(x)
    total += correlate_freqs(x, benign_or_malignant[:mid])

total = total / len(set_of_predictions)

print( "Average error rate w/weights: " + str(1 - (total / len(benign_or_malignant[:mid]))) )
print( "Average error rate w/weights: " + str(total) )