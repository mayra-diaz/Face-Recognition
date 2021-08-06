import heapq
import math
import json
import face_recognition
from rtree import index

from util import *


def init_search(Q, source):
    complete_path = 'data/' + source + '/'
    number_of_files = get_number_of_files(source)

    with open("dict_" + source + ".json") as json_file:
        dict = json.load(json_file)

    first_picture = face_recognition.load_image_file(complete_path + Q)
    first_face_encoding = face_recognition.face_encodings(first_picture)

    if first_face_encoding:
        list_of_values = first_face_encoding[0].tolist()
        list_of_values = (get_point(list_of_values))
        property = index.Property()
        property.dimension = 128
        property.buffering_capacity = 3 + int(math.log(number_of_files, 10) ** 2)
        property.dat_extension = 'data'
        property.idx_extension = 'index'
        property.overwrite = False
        index_ = index.Index('face_recognition_index_' + source, properties=property)
        return index_, list_of_values, dict
    else:
        return None, None, None

def validate_range(neighbor_vector, range_vector):
    i = 0
    j = len(neighbor_vector) // 2
    while i < j:
        if not range_vector[i] <= neighbor_vector[i] <= range_vector[len(neighbor_vector) // 2 + i]:
            return False
        i+=1
    return True


def create_range_vector(vector, r):
    range_vector = [0] * len(vector)
    i = 0
    j = len(vector) // 2
    while i < j:
        range_vector[i] = vector[i] - r
        range_vector[i + len(vector) // 2] = vector[i] + r
        i+=1
    return range_vector

def KNN(Q, k, path):
    index_, vector, dict = init_search(Q, path)
    if index_ == None:
        return []
    pictures = index_.intersection(index.bounds, objects=True)
    neighbors = []
    for picture in pictures: 
        d = euclidean_distance(vector, picture.bbox)
        heapq.heappush(neighbors, (d * -1, picture.id))
        if k < len(neighbors):
            heapq.heappop(neighbors)
    neighbors = [(i, d * -1) for d, i in neighbors]
    neighbors.sort(key=lambda tup: tup[1])
    return [dict[str(i)] for i in neighbors]
        
def range_search(Q, r, path):
    index_, vector, dict = init_search(Q, path)
    if index_ == None:
        return []
    pictures = index_.intersection(index.bounds, objects=True)
    neighbors = []
    range_vector = create_range_vector(vector, r)
    for picture in pictures:
        if validate_range(picture.bbox, range_vector):
            neighbors.append(picture.id)
    return [dict[str(i)] for i in neighbors]