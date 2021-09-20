import csv
import math
from collections import Counter
import numpy as np
import pandas as pd
import random
from main import *
import matplotlib.pyplot as plt
import matplotlib.markers

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

def plot(actual, prediction, name, figure):

    r = {2:[],4:[]}
    num_correct = {2:0,4:0}
    for x in range(len(actual)):
        r[int(actual[x])].append(prediction[x])

        if(int(actual[x]) == prediction[x]):
            num_correct[int(actual[x])] += 1

    for x in r:
        print("class = " + str(x) + " percent correct: " + str(num_correct[x]/len(r[x])))

    # b = 4

    # plt.figure(figure)
    # plt.plot(list(range(0,len(r[b]))),[b]*len(r[b]),label = "expected for c = " + str(b),linewidth=3, color = 'hotpink',zorder=1)
    # plt.scatter(list(range(0,len(r[b]))),r[b],label = name + " for c = " + str(b), linewidths=1,zorder=2,color = 'black', marker=matplotlib.markers.TICKDOWN)

def unweighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element),  benign_or_malignant[mid + index]])
        index += 1

    total_distances.sort()

    distances_to_k = [total_distances[i] for i in range(k)]

    
    class_dict = {2:[],4:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)class_dict[i]:
        class_dict[int(distance[-1])].append(distance[0])

    #each index is associated with the number of nodes with that 
    #class where class = index in nearest neighbors

    new_list = {}

    for i in class_dict: 
        #Associate each class with the length of it's respective distance array
        new_list[(i)] = len(class_dict[i])

    return max(new_list, key=new_list.get)

def weighted_KNN_classification(test_set, element,k):
    total_distances = []
    
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance(x,element),  benign_or_malignant[mid + index]])
        index += 1

    total_distances.sort()

    distances_to_k = [total_distances[i] for i in range(k)]
    distances = [total_distances[i][0] for i in range(len(distances_to_k))]

    maximum = max(distances)
    minimum = min(distances)


    class_dict = {2:[],4:[]}

    for distance in distances_to_k:
        class_dict[int(distance[-1])].append(distance[0])

    #each index is associated with the sum of the weight for that class where class = index
    new_list = [0,0]

    c = 0
    for i in class_dict: 
        
        for d in class_dict[i]:

            if maximum != minimum:
                new_list[c] += ((maximum - d) / (maximum - minimum))
            else:
                new_list[c] += 1

        c += 1
    return new_list.index(max(new_list))*2 + 2


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


#___________________________RUNNING BELOW___________________________

mid = len(total_set)//2
k = 15

#________ UNWEIGHTED ________

set_of_predictions = []

predictions = []

for x in range(mid):

    predictions.append(unweighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))
set_of_predictions.append(predictions)

total = 0

avg_set_of_predict = [0] * len(benign_or_malignant[:mid])

#if needed can run multiple trails, and should automatically average
for x in set_of_predictions:

    print(count_correct(x, benign_or_malignant[:mid]))

    total += count_correct(x, benign_or_malignant[:mid])

    for e in range(len(x)):
        avg_set_of_predict[e] += x[e]

#average general prediction, num correct / expected correct, aka number of elements in set
total = total / len(set_of_predictions) 

#grab first prediction, with the first k, since only 1 k used in our implementation
#use this to average our predictions
#a little redudant, but can be extrapolated for further use in future
avg_set_of_predict = set_of_predictions[0]

plot(benign_or_malignant[:mid], avg_set_of_predict, "unweighted", 0)

print( "Average error rate w/o weights: " + str(1 - (total / len(benign_or_malignant[:mid]))) )
print( "Average correct rate w/o weights: " + str((total / len(benign_or_malignant[:mid]))) )


#__________ WEIGHTED __________

set_of_predictions = []

predictions = []

for x in range(mid):

    predictions.append(weighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

set_of_predictions.append(predictions)

total = 0

#grab first prediction, with the first k, since only 1 k used in our implementation
#use this to average our predictions
#a little redudant, but can be extrapolated for further use in future
avg_set_of_predict = [0] * len(benign_or_malignant[:mid])

#if needed can run multiple trails, and should automatically average
for x in set_of_predictions:
    total += count_correct(x, benign_or_malignant[:mid])
    for e in range(len(x)):
        avg_set_of_predict[e] += x[e]

#average general prediction, num correct / expected correct, aka number of elements in set
total = total / len(set_of_predictions)

for e in range(len(avg_set_of_predict)):
    avg_set_of_predict[e] = avg_set_of_predict[e] / len(set_of_predictions)

plot(benign_or_malignant[:mid], avg_set_of_predict, "weighted", 1)

print( "Average error rate w/weights: " + str(1 - (total / len(benign_or_malignant[:mid]))) )
print( "Average correct rate w/weights: " + str((total / len(benign_or_malignant[:mid]))) )

#if ploting uncomment plot lines in plot function, and uncomment below as well

# plt.show()





