import csv
import math
from collections import Counter
import numpy as np
import random

filename = "datasets/iris.data"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []
qualities = []

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

    random.shuffle(shuffled)

    for row in shuffled:

        # print(row)

        floated_row = [float(row[i]) for i in range(len(row)-1)]
        total_set.append(floated_row)
        # print(row)
        # print(row[-1])
        qualities.append(row[-1])

        # print(floated_row)



    for row in range(len(total_set)):
        normalized_set.append(normalization(total_set[row]))
    # print(normalized_set)


def euclidian_distance(row1, row2):
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)
    

mid = len(normalized_set)//2

list_of_ks = [15]

def unweighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x))],element),  qualities[mid + index]])
        index += 1

    total_distances.sort()

    total_distances = total_distances[1:]

    distances_to_k = [total_distances[i] for i in range(k)]

    #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
    above_dict = {0:[],1:[],2:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:

        above_dict[(classification(distance[-1]))].append(distance[0])

    new_list = {}

    for i in above_dict: 

        new_list[(i)] = len(above_dict[i])


    return max(new_list, key=new_list.get)


#first half is test
#second half we use classify

#list_of_ks = [1, 5, 10, 15, 30, 45, 55, 65, 75, 85, 95, 155, 165, 175]

# list_of_ks = [int(math.pow(len(qualities[:mid]), 1/2))]



set_of_predictions = []

for i in range(100): 

    predictions = []

    for x in range(mid):

        predictions.append(unweighted_KNN_classification(normalized_set[mid:], normalized_set[x], list_of_ks[0]))

    set_of_predictions.append(predictions)


# print(predictions)
# print(qualities[:mid])

total = 0

for x in set_of_predictions:
    # print(x)
    total += correlate_freqs(x, qualities[:mid])

total = total / len(set_of_predictions)

print( "Average error rate w/o weights: " + str(1 - (total / len(qualities))) )
print( "Average error rate w/o weights: " + str(total)) 

def weighted_KNN_classification(test_set, element,k):
    total_distances = []
    
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x))],element), qualities[mid + index]])
        index += 1

    total_distances.sort()
    total_distances = total_distances[1:]
    # print(total_distances)

    distances_to_k = [total_distances[i] for i in range(k)]
    distances = [total_distances[i][0] for i in range(len(distances_to_k))]

    #print(distances)
    maximum = max(distances)
    minimum = min(distances)

    # print(maximum)
    # print(minimum)

    #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
    above_dict = {0:[],1:[],2:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:
        above_dict[(classification(distance[-1]))].append(distance[0])

    new_list = [0,0,0]

    for i in above_dict: 
        # print(i)
        for d in above_dict[i]:

            if maximum != minimum:
                new_list[i] += ((maximum - d) / (maximum - minimum))
            else:
                new_list[i] += 1


    return new_list.index(max(new_list))
    

# list_of_ks = [int(math.pow(len(qualities[:mid]), 1/2))]

set_of_predictions = []

for i in range(100): 

    predictions = []

    for x in range(mid):

        predictions.append(weighted_KNN_classification(normalized_set[mid:], normalized_set[x], list_of_ks[0]))

    set_of_predictions.append(predictions)


total = 0

for x in set_of_predictions:
    # print(x)
    total += correlate_freqs(x, qualities[:mid])

total = total / len(set_of_predictions)

print( "Average error rate w/weights: " + str(1 - (total / len(qualities))) )
print( "Average error rate w/weights: " + str(total) )









