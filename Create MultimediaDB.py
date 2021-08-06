import face_recognition
from rtree import index
import os
import json
import math
from util import *

class RtreeIndex:
    def __init__(self, source, out='DB/'):

        self.total_files = get_number_of_files(source)
        self.name = source
        self.dict = {}

        p = index.Property()
        p.dimension = 128
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        p.buffering_capacity = int(math.log(self.total_files, 10) ** 2) + 3
        idx = index.Index(out+'index_' + self.name, properties=p)

        for filename in os.listdir(self.source):
            photo = face_recognition.load_image_file(self.source + filename)
            face_encoding = face_recognition.face_encodings(photo)
            if face_encoding:
                values = face_encoding[0].tolist()
                values = (get_point(values))
                self.dict.setdefault(len(self.dict), filename)
                idx.insert(len(self.dict) - 1, values)

        file = open(out+"dict_" + self.name + ".json", "w")
        json.dump(self.dict, file)
        file.close()