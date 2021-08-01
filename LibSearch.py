import face_recognition
from rtree import index
import json
from math import log10

from util import *

def init_search(Q, path,frontEnd):
    if frontEnd==False:
        total_path = 'data/' + path + '/'
    else:
        total_path = 'data/imageInput/'
    total_files = int(path)

    with open("diccionario_" + path + ".json") as json_file:
        dict = json.load(json_file)

    picture_1 = face_recognition.load_image_file(total_path + Q)
    face_encoding_1 = face_recognition.face_encodings(picture_1)
    if face_encoding_1:
        values = face_encoding_1[0].tolist()
        values = (generate_point(values))
        p = index.Property()
        p.dimension = 128  # D
        p.buffering_capacity = int(log10(total_files) ** 2) + 3 # M
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        p.overwrite = False
        idx = index.Index('face_recognition_index_' + path, properties=p)
        return idx, values, dict
    else:
        return None,None,None


def KNN_FaceRecognition(Q, k, path,frontEnd=True):
    idx, feature_vector, names_dict = init_search(Q, path,frontEnd)
    if idx == None:
        return []
    temp_vecinos = list(idx.nearest(coordinates=feature_vector, num_results=k))
    vecinos = []
    for i in temp_vecinos:
        vecinos.append(names_dict[str(i)])
    return vecinos


#######################################

def generate_range_vector(feature_vector, r):
    range_vector = [0] * len(feature_vector)
    for i in range(len(feature_vector) // 2):
        range_vector[i] = feature_vector[i] - r
        range_vector[i + len(feature_vector) // 2] = feature_vector[i] + r
    return range_vector

def range_search_rtree(Q, r, path,frontEnd):
    idx, feature_vector, names_dict = init_search(Q, path,frontEnd)
    result = []
    range_vector = generate_range_vector(feature_vector, r)
    temp_vecinos = idx.intersection(range_vector, objects=True)
    cont = 0
    for vecino in temp_vecinos:
        result.append(names_dict[str(vecino.id)])
    return result