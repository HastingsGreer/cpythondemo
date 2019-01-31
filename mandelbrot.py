import spam
import cv2
import numpy as np


minx, miny, maxx, maxy = -2, -2, 2, 2

def left(a, b):
    return a, (a + b) / 2
def right(a, b):
    return (a + b) / 2, b
def middle(a, b):
    return 3 * a / 4 + b / 4, a / 4 + 3 * b / 4

while True:
    x, y = np.mgrid[miny:maxy:500j, minx:maxx:500j]
    z = y * 0
    spam.numpyex(x, y, z)
    cv2.imshow("h", ((z % 256)).astype(np.uint8))
    k = cv2.waitKey(9000)
    k = chr(k) if k != -1 else "none"
    print(k)
    if k == 'q':
        minx, maxx = left(minx, maxx)
        miny, maxy = left(miny, maxy)
    if k == "e":
        minx, maxx = right(minx, maxx)
        miny, maxy = left(miny, maxy)
    if k == "z":
        minx, maxx = left(minx, maxx)
        miny, maxy = right(miny, maxy)
    if k == "c":
        minx, maxx = right(minx, maxx)
        miny, maxy = right(miny, maxy)
    if k == "s":
        minx, maxx = middle(minx, maxx)
        miny, maxy = middle(miny, maxy)

    if k == "w":
        minx, maxx = middle(minx, maxx)
        miny, maxy = left(miny, maxy)
    if k == "x":
        minx, maxx = middle(minx, maxx)
        miny, maxy = right(miny, maxy)
    if k == "d":
        minx, maxx = right(minx, maxx)
        miny, maxy = middle(miny, maxy)
    if k == "a":
        minx, maxx = left(minx, maxx)
        miny, maxy = middle(miny, maxy)
    if k == "p":
        d = (maxx - minx) / 2
        minx, maxx = minx - d, maxx + d
        d = (maxy - miny) / 2
        miny, maxy = miny - d, maxy + d
