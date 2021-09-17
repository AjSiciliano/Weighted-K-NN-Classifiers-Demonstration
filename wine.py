import csv
import math
from collections import Counter

filename = "datasets/winequality-red.csv"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
qualities = []


def correlate_freqs(v1,v2):

    numerator = 0
    v1_mag = 0
    v2_mag = 0

    for i in range(len(v1)):
        numerator += ( v1[i] * v2[(i)%len(v2)] )
        v1_mag += math.pow(v1[i], 2)
        v2_mag += math.pow(v2[(i)%len(v2)], 2)

    v1_mag = math.sqrt(v1_mag)
    v2_mag = math.sqrt(v2_mag)

    return numerator / (v1_mag * v2_mag)


def normalization(array):
    maximum = max(array)
    minimum = min(array)
    normalized_list = []
    for element in array:
        e = element
        normalized_list.append((e - minimum) / (maximum - minimum))

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
        total_set.append(normalization(floated_row))

def euclidian_distance(row1, row2):
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)
    
def nearest_neighbors(element, k):
    distances = []
    index = 0
    for x in test_set:
    
        distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element), index,  qualities[index]])
        index += 1

    distances.sort()
    #excludes element in search and adds new nearest neighbor
    print([distances[i] for i in range(k+1)[1:k+1]])

    return [distances[i][2] for i in range(k)]

# NN = nearest_neighbors(test_set[8], 3)

# print(Counter(NN).most_common(1))


def weighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element),  qualities[index]])
        index += 1

    total_distances.sort()
    #distances_to_k = [total_distances[i] for i in range(k+1)[1:k+1]]
    distances_to_k = [total_distances[i] for i in range(k)]

    #we do the implementation below because for any k value, we need to keep track of the classes that appear to be our NN
    above_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}

    for distance in distances_to_k:
        #if distance[-1] == (float)above_dict[i]:
        above_dict[int(distance[-1])].append(distance[0])

    #above_dict_trimmed = filter(None, above_dict) #empty list

    #line of code below removes all empty lists from above_dict (syntax may be a little wrong. Maybe find another way to do it)

    #above_dict(filter(lambda x: x, lst))
    

    #this is now the new above_dict-> {5:[0.054, 0.054],8:{0.057}}

    #START of sum of unweighted distances

    new_list = {} #which will contain class and total sum of distance

    for i in above_dict: 
        s = 0

        for element in above_dict[i]:
    
            s += element

        new_list[(i)] = s

    return max(new_list, key=new_list.get)

    #END of sum of unweighted distances

    #new_list is now {5:[0.108],8:[0.057]}
    #how to return the class that has the max?? 
    # maximum = 0
    # for i in new_list:
    #     for distance in new_list[i]:
    #         if distance > max
    #         max = distance

    # return new_list[i][max] #idk if the syntax is correct but it will retrieve the class with the updated max

   

    #return -1


#first half is test
#second half we use classify

list_of_ks = [1, 5, 10, 15, 30, 45, 55, 65, 75, 500]



mid = len(total_set)//2

set_of_predictions = []

for k in list_of_ks:

    predictions = []

    for x in range(mid):

        predictions.append(weighted_KNN_classification(total_set[mid:], total_set[x], k))

    set_of_predictions.append(predictions)


# print(predictions)
# print(qualities[:mid])
for x in set_of_predictions:
    print(correlate_freqs(x, qualities[:mid]))

#     for x in range((len(total_set) - 1) / 2) :
        

#     classifications_per_k.append(appended)


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