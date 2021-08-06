import face_recognition
from rtree import index
import json
from math import log10

from util import *

def init_search(Q, path,frontEnd):
    if not frontEnd:
        complete_path = 'data/' + path + '/'
    else:
        complete_path = 'data/imageInput/'
    number_of_files = get_number_of_files(path)

    with open("dict_" + path + ".json") as json_file:
        dict = json.load(json_file)

    first_picture = face_recognition.load_image_file(complete_path + Q)
    first_face_encoding = face_recognition.face_encodings(first_picture)
    if first_face_encoding:
        list_of_values = first_face_encoding[0].tolist()
        list_of_values = (get_point(list_of_values))
        property = index.Property()
        property.dimension = 128
        property.buffering_capacity = 3 + int(log10(number_of_files) ** 2)
        property.dat_extension = 'data'
        property.idx_extension = 'index'
        property.overwrite = False
        index_ = index.Index('face_recognition_index_' + path, properties=property)
        return index_, list_of_values, dict
    else:
        return None, None, None

def create_range_vector(vector, r):
    range_vector = [0] * len(vector)
    i = 0
    j = len(vector) // 2
    while i < j:
        range_vector[i] = vector[i] - r
        range_vector[i + len(vector) // 2] = vector[i] + r
        i+=1
    return range_vector

def KNN(Q, k, path, frontEnd=True):
    index_, list_of_values, dict = init_search(Q, path, frontEnd)
    if index_ == None:
        return []
    aux_neighbor = list(index_.nearest(coordinates=list_of_values, num_results=k))
    neighbor = []
    for i in aux_neighbor:
        neighbor.append(dict[str(i.id)])
    return neighbor

def range_search(Q, r, path,frontEnd):
    index_, list_of_values, dict = init_search(Q, path, frontEnd)
    ans = []
    range_vector = create_range_vector(list_of_values, r)
    aux_neighbor = index_.intersection(range_vector, objects=True)
    for i in aux_neighbor:
        ans.append(dict[str(i.id)])
    return ans