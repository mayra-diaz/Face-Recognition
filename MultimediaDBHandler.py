import face_recognition
from rtree import index
from util import *
import os
import json
import math

dict = {}

class RtreeIndex:

    def __init__(self, path):
        numberOfFiles = int(path)
        completePath = 'data/' + path + '/'

        p = index.Property()
        p.dimension = 128
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        p.buffering_capacity = int(math.log(numberOfFiles, 10) ** 2) + 3
        idx = index.Index('face_recognition_index_' + path, properties=p)

        self.numberOfFiles = numberOfFiles
        self.path = path
        self.completePath = completePath

        for filename in os.listdir(completePath):
            picture_1 = face_recognition.load_image_file(completePath + filename)
            face_encoding_1 = face_recognition.face_encodings(picture_1)
            if face_encoding_1:
                values = face_encoding_1[0].tolist()
                values = (get_point(values))
                dict.setdefault(len(dict), filename)
                idx.insert(len(dict) - 1, values)

        file = open("diccionario_" + path + ".json", "w")
        json.dump(dict, file)
        file.close()