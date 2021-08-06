import face_recognition
from rtree import index
from util import *
import os
import json
import math

dict = {}

class RtreeIndex:

    def __init__(self, path):
        number_of_files = int(path)
        complete_path = 'data/' + path + '/'

        p = index.Property()
        p.dimension = 128
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        p.buffering_capacity = int(math.log(number_of_files, 10) ** 2) + 3
        idx = index.Index('face_recognition_index_' + path, properties=p)

        self.number_of_files = number_of_files
        self.path = path
        self.complete_path = complete_path

        for filename in os.listdir(complete_path):
            picture_1 = face_recognition.load_image_file(complete_path + filename)
            face_encoding_1 = face_recognition.face_encodings(picture_1)
            if face_encoding_1:
                values = face_encoding_1[0].tolist()
                values = (get_point(values))
                dict.setdefault(len(dict), filename)
                idx.insert(len(dict) - 1, values)

        file = open("diccionario_" + path + ".json", "w")
        json.dump(dict, file)
        file.close()