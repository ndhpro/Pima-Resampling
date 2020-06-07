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
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return np.array(neighbors, dtype=np.float)


def rbu(file_name=sys.argv[1]):
    data = pd.read_csv(file_name)
    data['index'] = range(len(data))
    print(len(data))
    n = abs(len(data[data["' Outcome'"]==0]) - len(data[data["' Outcome'"]==1]))
    borders = list()
    for test_row in data.values:
        if test_row[-2] == 0:
            neighbors = get_neighbors(data.values, test_row, 10)
            m = len([i for i in neighbors if i[-2]==1])
            if m >= 5:
                borders.append(test_row[-1])
    borders = choices(borders, k=n)
    data = data.drop(borders)
    print(len(data))
    data.iloc[:, :-1].to_csv('data/rbu.csv', index=None)
    

if __name__ == "__main__":
    rbu()