import math
import random
import numpy as np

def getRadian(x1, y1, x2, y2):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    vec = b - a
    #引数の順番がx, yではなくy, xなので注意
    return math.atan2(vec[1], vec[0])


if __name__ == '__main__' :
    a = np.array([1, 1])
    b = np.array([2, 2])

    p = math.pi
    radian = math.radians(90)
    #print(math.degrees(p))
    #print(math.degrees(radian))
    #print(math.cos(radian), math.sin(radian))

    #print(math.radians(180))

    x1 = 19
    y1 = 323
    x2 = 300
    y2 = 300
    print(math.degrees(getRadian(x1, y1, x2, y2)))

