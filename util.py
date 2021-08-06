import os

def get_point(vector):
    doble = vector + vector
    return tuple(i and i for i in doble)

def euclidean_distance(a, b):
    return sum([(a[i]-b[i])**2 for i in range(len(a))])**0.5

def get_number_of_files(directory):
    files_n = 0
    for base, dirs, files in os.walk(directory):
        for f in files:
            files_n += 1
    return files_n