
def get_point(vector):
    doble = vector + vector
    return tuple(i and i for i in doble)

def euclidean_distance(a, b):
    return sum([(a[i]-b[i])**2 for i in range(len(a))])**0.5