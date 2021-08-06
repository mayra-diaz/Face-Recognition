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
        return None

def validate_range(neighbor_vector, range_vector):
    for i in range(len(neighbor_vector) // 2):
        if not range_vector[i] <= neighbor_vector[i] <= range_vector[len(neighbor_vector) // 2 + i]:
            return False
    return True

def generate_range_vector(feature_vector, r):
    range_vector = [0] * len(feature_vector)
    for i in range(len(feature_vector) // 2):
        range_vector[i] = feature_vector[i] - r
        range_vector[i + len(feature_vector) // 2] = feature_vector[i] + r
    return range_vector

def KNN_sequential(Q, k, path):
    idx, feature_vector, names_dict = init_search(Q, path)
    if idx == None:
        return []
    images = idx.intersection(idx.bounds, objects=True) # puntero al indice
    neighbors = []
    for image in images: # recorrer bloque a bloque
        d = euclidean_distance(feature_vector, image.bbox)
        heapq.heappush(neighbors, (-d, image.id))
        if len(neighbors) > k:
            heapq.heappop(neighbors)
    neighbors = [(i, d * -1) for d, i in neighbors]
    neighbors.sort(key=lambda tup: tup[1])
    return [names_dict[str(i)] for i, d in neighbors]
        
def range_search_sequential(Q, r, path):
    idx, feature_vector, names_dict = init_search(Q, path)
    if idx == None:
        return []
    images = idx.intersection(idx.bounds, objects=True) # puntero al indice
    neighbors = []
    range_vector = generate_range_vector(feature_vector, r)
    for image in images: # recorrer bloque a bloque
        if validate_range(image.bbox, range_vector):
            neighbors.append(image.id)
    return [names_dict[str(i)] for i in neighbors]