import csv
import math
from collections import Counter
import pandas as pd

# initializing the titles and rows list
benign_or_malignant = []

def normalization(array):
    maximum = max(array)
    minimum = min(array)
    normalized_list = []
    for element in array:
        e = element
        normalized_list.append((e - minimum) / (maximum - minimum))

    return normalized_list

df = pd.read_csv('/Users/chriswu/Downloads/breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
#drop unnecessary columns (id and class) ...also test without drop
df.drop(['id'],1,inplace=True)

#df has random quotes. Convert all to float
full_data = df.astype(float).values.tolist()

for row in full_data:
    benign_or_malignant.append(row.pop())

def euclidian_distance(row1, row2):
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)
    
def nearest_neighbors(element, k):
    distances = []
    index = 0
    for x in full_data:
        distances.append([euclidian_distance(x,element),  benign_or_malignant[index]])
        index+=1
    distances.sort()
    #excludes element in search and adds new nearest neighbor
    print([distances[i] for i in range(k+1)[1:k+1]])

    return [distances[i][1] for i in range(k)]

NN = nearest_neighbors(full_data[8], 3)

print(Counter(NN).most_common(1))