import csv
import math
from collections import Counter
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.markers

filename = "datasets/winequality-red.csv"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []
qualities = []

def count_correct(v1,v2):
    #Count the number of similarities between v1 and v2

    num_correct = 0

    for i in range(len(v1)):
        
        if (v1[i] == v2[i]):
            num_correct += 1

    return num_correct

def normalization(array):
    #Normalize the attributes in respect to eachother in each column
    normalized_list = []

    maximum = np.maximum.reduce(total_set)
    minimum = np.minimum.reduce(total_set)

    for i in range(len(array)):
        e = array[i]

        normalized_list.append((e - minimum[i]) / (maximum[i] - minimum[i]))

    return normalized_list

def euclidian_distance(row1, row2):
    #Calculate euclidian distance between two vectors r1 and r2
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)

def unweighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element),  qualities[mid + index]])
        index += 1

    total_distances.sort()

    distances_to_k = [total_distances[i] for i in range(k)]

    class_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

    for distance in distances_to_k:

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
        total_distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element),  qualities[mid + index]])
        index += 1

    total_distances.sort()
    
    distances_to_k = [total_distances[i] for i in range(k)]
    distances = [total_distances[i][0] for i in range(len(distances_to_k))]

    maximum = max(distances)
    minimum = min(distances)

    class_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

    for distance in distances_to_k:

        class_dict[int(distance[-1])].append(distance[0])

    #each index is associated with the sum of the weight for that class where class = index
    new_list = [0,0,0,0,0,0,0,0,0,0,0]

    for i in class_dict: 

        for d in class_dict[i]:

            if maximum != minimum:
                new_list[i] += ((maximum - d) / (maximum - minimum)) #weights per element formula
            else:
                new_list[i] += 1

    return new_list.index(max(new_list))


#method used to plot example graphs and find figures for individual attributes
def plot(actual, prediction, name, figure):

    r = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

    num_correct = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}

    for x in range(len(actual)):
        r[int(actual[x])].append(prediction[x])

        if(int(actual[x]) == prediction[x]):
            num_correct[int(actual[x])] += 1

    for x in r:
        if (len(r[x]) != 0):
            print("class = " + str(x) + " percent correct: " + str(num_correct[x]/len(r[x])))

    # b = 9 #-> the class number associated to graph

    # plt.figure(figure)
    # plt.plot(list(range(0,len(r[b]))),[b]*len(r[b]),label = "expected for c = " + str(b),linewidth=3, color = 'hotpink',zorder=1)
    # plt.scatter(list(range(0,len(r[b]))),r[b],label = name + " for c = " + str(b), linewidths=1,zorder=2,color = 'black', marker=matplotlib.markers.TICKDOWN)


#___________________________IMPLEMENTATION BELOW___________________________

#read and splice the data
#https://www.geeksforgeeks.org/working-csv-files-python/
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    next(csvreader)

    shuffled = []

    for row in csvreader:

        shuffled.append(row)

    random.seed(4) #Seed for replication

    random.shuffle(shuffled)

    for row in shuffled:
        spliced_row = row[0].split(";")
        floated_row = [float(i) for i in spliced_row]
        qualities.append(floated_row.pop())
        total_set.append(floated_row)

    for row in range(len(total_set)):
        normalized_set.append(normalization(total_set[row]))

#get the middle index for splitting the data into two sections
#first half is test
#second half we use classify
mid = len(normalized_set)//2

list_of_ks = [int(math.pow(len(qualities[:mid]), 1/2))] #add more k's to get average of multiple k's

#________ UNWEIGHTED ________

set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(unweighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

    set_of_predictions.append(predictions)

total = 0

avg_set_of_predict = [0] * len(qualities[:mid])

for x in set_of_predictions:
    total += count_correct(x, qualities[:mid])
    for e in range(len(x)):
        avg_set_of_predict[e] += x[e]

#average general prediction, num correct / expected correct, aka number of elements in set
total = total / len(set_of_predictions) 

#grab first prediction, with the first k, since only 1 k used in our implementation
#use this to average our predictions
#a little redudant, but can be extrapolated for further use in future
avg_set_of_predict = set_of_predictions[0]

plot(qualities[:mid], avg_set_of_predict, "unweighted", 0)

print( "Average error rate w/o weights: " + str(1 - (total / len(qualities[:mid]))) )
print( "Average correct rate w/o weights: " + str((total / len(qualities[:mid]))) )


#__________ WEIGHTED __________

set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(weighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

    set_of_predictions.append(predictions)

total = 0

#grab first prediction, with the first k, since only 1 k used in our implementation
#use this to average our predictions
#a little redudant, but can be extrapolated for further use in future
avg_set_of_predict = [0] * len(qualities[:mid])

for x in set_of_predictions:
    # print(x)
    total += count_correct(x, qualities[:mid])
    for e in range(len(x)):
        avg_set_of_predict[e] += x[e]

#average general prediction, num correct / expected correct, aka number of elements in set
total = total / len(set_of_predictions)

for e in range(len(avg_set_of_predict)):
    avg_set_of_predict[e] = avg_set_of_predict[e] / len(set_of_predictions)


plot(qualities[:mid], avg_set_of_predict, "weighted", 1)

print( "Average error rate w/weights: " + str(1 - (total / len(qualities[:mid]))) )
print( "Average correct rate w/weights: " + str((total / len(qualities[:mid]))) )



#if ploting uncomment plot lines in plot function, and uncomment below as well

# leg = plt.legend()

# leg_lines = leg.get_lines()

# plt.show()

