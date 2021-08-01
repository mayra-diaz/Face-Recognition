import heapq
import math
import json
import face_recognition
from rtree import index

from util import *


def init_search(Q, path):
    total_path = 'data/' + path + '/'
    total_files = int(path)

    with open("diccionario_" + path + ".json") as json_file:
        dict = json.load(json_file)

    picture_1 = face_recognition.load_image_file(total_path + Q)
    face_encoding_1 = face_recognition.face_encodings(picture_1)
    if face_encoding_1:
        values = face_encoding_1[0].tolist()
        values = (get_point(values))
        p = index.Property()
        p.dimension = 128  # D
        p.buffering_capacity = int(math.log(total_files, 10) ** 2) + 3 # M
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        p.overwrite = False
        idx = index.Index('face_recognition_index_' + path, properties=p)
        return idx, values, dict
    else:
        return None

def is_inside_range(neighbor_feature_vector, range_vector):
    for i in range(len(neighbor_feature_vector) // 2):
        if not range_vector[i] <= neighbor_feature_vector[i] <= range_vector[i + len(neighbor_feature_vector) // 2]:
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
        if is_inside_range(image.bbox, range_vector):
            neighbors.append(image.id)
    return [names_dict[str(i)] for i in neighbors]