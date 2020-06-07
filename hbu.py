import sys
import numpy as np
import pandas as pd
from math import sqrt
from random import choices


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-2):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    i = 0
    while len(neighbors) < num_neighbors and i < len(distances):
        if distances[i][0][-2] == 0:
            neighbors.append(distances[i][0][-1])
        i += 1
    return neighbors


def get_margin(train, test_row):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    for train_row, dist in distances:
        if test_row[-2] == train_row[-2]:
            nearesthit = train_row
            break
    for train_row, dist in distances:
        if test_row[-2] != train_row[-2]:
            nearestmiss = train_row
            break
    margin = 1/2 * (euclidean_distance(test_row, nearestmiss) - euclidean_distance(test_row, nearesthit))
    return margin


def hbu(file_name=sys.argv[1]):
    data = pd.read_csv(file_name)
    data['index'] = range(len(data))
    data_ = pd.read_csv(file_name)
    data_['index'] = range(len(data_))
    print(len(data))
    N = len(data[data["' Outcome'"]==0])
    p = len(data[data["' Outcome'"]==1])
    n = N - p
    k = 10
    max_mar = -1e9
    min_mar = 1e9
    for test_row in data.values:
        if test_row[-2] == 1:
            max_mar = max(max_mar, get_margin(data.values, test_row))
            min_mar = min(min_mar, get_margin(data.values, test_row))

    for test_row in data_.values:
        if test_row[-2] == 1:
            nos = int(n/p * (k + (max_mar-get_margin(data_.values, test_row))*(max_mar-min_mar)))
            neighbors = get_neighbors(data.values, test_row, nos)
            data = data.drop(neighbors)
            n -= nos
            p -= 1
            if n <= 0:
                break
    print(len(data))
    data.iloc[:, :-1].to_csv('hbu.csv', index=None)
    

if __name__ == "__main__":
    hbu()
