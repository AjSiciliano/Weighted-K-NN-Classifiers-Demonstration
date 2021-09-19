import csv
import math
from collections import Counter
import numpy as np
import random

filename = "datasets/winequality-red.csv"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []
qualities = []


def correlate_freqs(v1,v2):

    num_correct = 0

    for i in range(len(v1)):
        
        if (v1[i] == v2[i]):
            num_correct += 1
            # print("YAY")

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
  
    # extracting each data row one by one
    for row in csvreader:
        spliced_row = row[0].split(";")
        floated_row = [float(i) for i in spliced_row]
        qualities.append(floated_row.pop())
        total_set.append(floated_row)

    

    # print(total_set)

    for row in range(len(total_set)):
        #print(total_set[row])
        normalized_set.append(normalization(total_set[row]))
        # print(normalized_set[row])
        # print("___")

# print(normalized_set)
# print(total_set)
# print(len(normalized_set))
# print(len(total_set))

def euclidian_distance(row1, row2):
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)
    
# def nearest_neighbors(element, k):
#     distances = []
#     index = 0
#     for x in test_set:
    
#         distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element), index,  qualities[index]])
#         index += 1

#     distances.sort()
#     #excludes element in search and adds new nearest neighbor
#     print([distances[i] for i in range(k+1)[1:k+1]])

#     return [distances[i][2] for i in range(k)]

# NN = nearest_neighbors(test_set[8], 3)

# print(Counter(NN).most_common(1))

mid = len(normalized_set)//2


def unweighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element),  qualities[mid + index]])
        index += 1

    total_distances.sort()

    distances_to_k = [total_distances[i] for i in range(k)]

    #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
    above_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:
        above_dict[int(distance[-1])].append(distance[0])

    new_list = {}

    for i in above_dict: 

        new_list[(i)] = len(above_dict[i])


    return max(new_list, key=new_list.get)


#first half is test
#second half we use classify

#list_of_ks = [1, 5, 10, 15, 30, 45, 55, 65, 75, 85, 95, 155, 165, 175]

# list_of_ks = [int(math.pow(len(qualities[:mid]), 1/2))]

list_of_ks = [77]


set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(unweighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

    set_of_predictions.append(predictions)


# print(predictions)
# print(qualities[:mid])

total = 0

for x in set_of_predictions:
    # print(x)
    total += correlate_freqs(x, qualities[:mid])

total = total / len(set_of_predictions)

print( "Average error rate w/o weights: " + str(1 - (total / len(qualities[:mid]))) )
print( "Average error rate w/o weights: " + str(total)) 

def weighted_KNN_classification(test_set, element,k):
    total_distances = []
    
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element), qualities[mid + index]])
        index += 1

    total_distances.sort()
    # print(total_distances)

    distances_to_k = [total_distances[i] for i in range(k)]
    distances = [total_distances[i][0] for i in range(len(distances_to_k))]

    #print(distances)
    maximum = max(distances)
    minimum = min(distances)

    # print(maximum)
    # print(minimum)

    #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
    above_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:
        above_dict[int(distance[-1])].append(distance[0])

    new_list = [0,0,0,0,0,0,0,0,0,0,0]

    for i in above_dict: 
        # print(i)
        for d in above_dict[i]:

            if maximum != minimum:
                new_list[i] += ((maximum - d) / (maximum - minimum))
            else:
                new_list[i] += 1


    return new_list.index(max(new_list))
    

# list_of_ks = [int(math.pow(len(qualities[:mid]), 1/2))]

mid = len(normalized_set)//2

set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(weighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

    set_of_predictions.append(predictions)


# print(predictions)
# print(qualities[:mid])

total = 0

for x in set_of_predictions:
    # print(x)
    total += correlate_freqs(x, qualities[:mid])

total = total / len(set_of_predictions)

print( "Average error rate w/weights: " + str(1 - (total / len(qualities[:mid]))) )
print( "Average error rate w/weights: " + str(total) )



#     distances_to_k = [total_distances[i] for i in range(k)]


#     maximum = max(distances_to_k)
#     minimum = min(distances_to_k)

#     #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
#     above_dict = {0:[],1:[],2:[],3:[],4:[],5:[0.057,0.067],6:[],7:[],8:[0.082],9:[],10:[]}
# #5:[w] 8:[z]

    

#     for distance in distances_to_k:
 
#         above_dict[int(distance[-1])].append(distance[0])

# #
#     new_list = {} #which will contain class and total sum of distance

#     for i in above_dict: 
#         s = 0

#         for element in above_dict[i]:
    
#             s += element

#         new_list[(i)] = s

#     return max(new_list, key=new_list.get)






#potentially choosing the optimal K if we have time

# classifications_per_k = []

# for k in range((len(total_set) - 1) / 2):
#     appended = []

#     for x in range((len(total_set) - 1) / 2) :
        

#     classifications_per_k.append(appended)



#
#
#/// WEIGHTED CALCULATION PORTION
#  this is the new above_dict -> {5:[0.054, 0.054],8:{0.057}}
#for i in above_dict:
#   maximum = max(above_dict[i])
#   minimum = min(above_dict[i])
#
#   new_list = [] which will contain class and total sum of distance
#
#   for element in above_dict[i]:
#   weighted = (maximum - element)/(maximum-minimum)
# 
#   new_list.append([i],weighted])
#
#maximum = 0
#for i in new_list:
#   for distance in new_list[i]:
#       if distance > max
#       max = distance
#   return new_list[i][max] idk the syntax but it will retrieve the class with the updated max
#return -1