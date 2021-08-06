import face_recognition
from rtree import index
import os
import json
import math

diccionario = {}

class RtreeIndex:

    def __init__(self, path):
        total_files = int(path)
        total_path = 'data/' + path + '/'

        p = index.Property()
        p.dimension = 128
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        p.buffering_capacity = int(math.log(total_files, 10) ** 2) + 3
        idx = index.Index('face_recognition_index_' + path, properties=p)

        self.total_files = total_files
        self.path = path
        self.total_path = total_path

        for filename in os.listdir(total_path):
            picture_1 = face_recognition.load_image_file(total_path + filename)
            face_encoding_1 = face_recognition.face_encodings(picture_1)
            if face_encoding_1:
                values = face_encoding_1[0].tolist()
                values = (generate_point(values))
                diccionario.setdefault(len(diccionario), filename)
                idx.insert(len(diccionario) - 1, values)

        file = open("diccionario_" + path + ".json", "w")
        json.dump(diccionario, file)
        file.close()