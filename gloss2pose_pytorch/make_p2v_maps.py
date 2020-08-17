import os
import json
from PIL import Image
import numpy as np
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt

def draw_subpart(xx, yy, hm, val):
    for x, y in zip(xx, yy):
        if x > 0.0 and y > 0.0:
                x = int(round(x))
                y = int(round(y))

                hm[y-2:y+2, x-2:x+2] = val

    for i in range(len(xx) - 1):
        x = [xx[i], xx[i+1]]
        y = [yy[i], yy[i + 1]]
        try:
            inter1 = interp1d(x, y, kind='slinear')

            for j in range(x[0], x[1] + 1):
                k = inter1(j)
                if not np.isnan(k):
                    kk = round(int(k))
                    hm[int(kk)][int(j)] = val
        except:
            print("interp1d failed")

        try:
            inter2 = interp1d(y, x, kind='slinear')

            for j in range(y[0], y[1] + 1):
                k = inter2(j)
                if not np.isnan(k):
                    kk = round(int(k))
                    hm[int(j)][int(kk)] = val
        except:
            print("interp1d failed")

        x = [xx[i + 1], xx[i]]
        y = [yy[i + 1], yy[i]]

        try:
            inter1 = interp1d(x, y, kind='slinear')

            for j in range(x[0], x[1] + 1):
                k = inter1(j)
                if not np.isnan(k):
                    kk = round(int(k))
                    hm[int(kk)][int(j)] = val
        except:
            print("interp1d failed")

        try:
            inter2 = interp1d(y, x, kind='slinear')

            for j in range(y[0], y[1] + 1):
                k = inter2(j)
                if not np.isnan(k):
                    kk = round(int(k))
                    hm[int(j)][int(kk)] = val
        except:
            print("interp1d failed")
    return hm

def draw_face(xf, yf, val):
    heat_map = np.zeros((1080, 1920), dtype=np.uint8)

    #contour
    x_c = [int(round(elem)) for elem in xf[0:17]]
    y_c = [int(round(elem)) for elem in yf[0:17]]

    heat_map = draw_subpart(x_c, y_c, heat_map, val)
    val += 1

    #eyebrow left
    x_el = [int(round(elem)) for elem in xf[18:22]]
    y_el = [int(round(elem)) for elem in yf[18:22]]

    heat_map = draw_subpart(x_el, y_el, heat_map, val)

    # eyebrow right
    x_er = [int(round(elem)) for elem in xf[23:27]]
    y_er = [int(round(elem)) for elem in yf[23:27]]

    heat_map = draw_subpart(x_er, y_er, heat_map, val)
    val += 1

    # nose upper
    x_nu = [int(round(elem)) for elem in xf[28:31]]
    y_nu = [int(round(elem)) for elem in yf[28:31]]

    heat_map = draw_subpart(x_nu, y_nu, heat_map, val)
    # nose lower
    x_nl = [int(round(elem)) for elem in xf[32:36]]
    y_nl = [int(round(elem)) for elem in yf[32:36]]

    heat_map = draw_subpart(x_nl, y_nl, heat_map, val)
    val += 1

    # eye left
    x_eyel = [int(round(elem)) for elem in xf[37:42]]
    y_eyel = [int(round(elem)) for elem in yf[37:42]]

    heat_map = draw_subpart(x_eyel, y_eyel, heat_map, val)

    # eye right
    x_eyer = [int(round(elem)) for elem in xf[43:48]]
    y_eyer = [int(round(elem)) for elem in yf[43:48]]

    heat_map = draw_subpart(x_eyer, y_eyer, heat_map, val)

    # eyeball left
    x_ebl = int(round(xf[68]))
    y_ebl = int(round(yf[68]))

    heat_map[y_ebl][x_ebl] = val

    # eyeball right
    x_ebr = int(round(xf[69]))
    y_ebr = int(round(yf[69]))

    heat_map[y_ebr][x_ebr] = val

    val += 1

    # mouth outer
    x_mo = [int(round(elem)) for elem in xf[49:60]]
    y_mo = [int(round(elem)) for elem in yf[49:60]]

    heat_map = draw_subpart(x_mo, y_mo, heat_map, val)

    # mouth inner
    x_mi = [int(round(elem)) for elem in xf[61:68]]
    y_mi = [int(round(elem)) for elem in yf[61:68]]

    heat_map = draw_subpart(x_mi, y_mi, heat_map, val)
    val += 1

    return val, heat_map

def draw_hand(xh, yh, val, heat_map):

    # thumb
    x_t = [int(round(elem)) for elem in xh[0:5]]
    y_t = [int(round(elem)) for elem in yh[0:5]]

    heat_map = draw_subpart(x_t, y_t, heat_map, val)

    # index
    x_i = [int(round(elem)) for elem in [xh[i] for i in (0, 5, 6, 7, 8)]]
    y_i = [int(round(elem)) for elem in [yh[i] for i in (0, 5, 6, 7, 8)]]

    heat_map = draw_subpart(x_i, y_i, heat_map, val)

    # middle
    x_m = [int(round(elem)) for elem in [xh[i] for i in (0, 9, 10, 11, 12)]]
    y_m = [int(round(elem)) for elem in [yh[i] for i in (0, 9, 10, 11, 12)]]

    heat_map = draw_subpart(x_m, y_m, heat_map, val)

    # ring
    x_r = [int(round(elem)) for elem in [xh[i] for i in (0, 13, 14, 15, 16)]]
    y_r = [int(round(elem)) for elem in [yh[i] for i in (0, 13, 14, 15, 16)]]

    heat_map = draw_subpart(x_r, y_r, heat_map, val)

    # pinky
    x_p = [int(round(elem)) for elem in [xh[i] for i in (0, 17, 18, 19, 20)]]
    y_p = [int(round(elem)) for elem in [yh[i] for i in (0, 17, 18, 19, 20)]]

    heat_map = draw_subpart(x_p, y_p, heat_map, val)
    return heat_map


#from https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/
def smoothListGaussian(list, degree=5):
    window = degree * 2 - 1

    weight = np.array([1.0] * window)

    weightGauss = []

    for i in range(window):
        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight

    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)

    return np.asarray(smoothed)


def load_data_from_numpy(path):
    data = np.load(path)
    X = data['input']
    xy_pose = X[:, 0:28]
    xy_face = X[:, 28:168]
    xy_hand_l = X[:, 168:210]
    xy_hand_r = X[:, 210:252]

    _x_p = xy_pose[:, ::2]
    _y_p = xy_pose[:, 1::2]

    _x_f = xy_face[:, ::2]
    _y_f = xy_face[:, 1::2]

    _x_hl = xy_hand_l[:, ::2]
    _y_hl = xy_hand_l[:, 1::2]

    _x_hr = xy_hand_r[:, ::2]
    _y_hr = xy_hand_r[:, 1::2]

    return _x_p, _y_p, _x_f, _y_f, _x_hl, _y_hl, _x_hr, _y_hr

def make_heatmaps(inputline, _fp):
    xy_pose = inputline[0:28]
    xy_face = inputline[28:168]
    xy_hand_l = inputline[168:210]
    xy_hand_r = inputline[210:252]

    _x_p = xy_pose[::2]
    _y_p = xy_pose[1::2]

    _x_f = xy_face[::2]
    _y_f = xy_face[1::2]

    _x_hl = xy_hand_l[::2]
    _y_hl = xy_hand_l[1::2]

    _x_hr = xy_hand_r[::2]
    _y_hr = xy_hand_r[1::2]

    heat_map = np.zeros((1080, 1920), dtype=np.uint8)
    val = 1
    for x, y in zip(_x_p, _y_p):
        if x > 0.0 and y > 0.0:
            x = int(round(x))
            y = int(round(y))

            heat_map[y - 10:y + 10, x - 10:x + 10] = val
        val += 1

    val, face_map = draw_face(_x_f, _y_f, val)

    hand_map = np.zeros((1080, 1920), dtype=np.uint8)
    hand_map = draw_hand(_x_hl, _y_hl, val, hand_map)
    val += 1
    hand_map = draw_hand(_x_hr, _y_hr, val, hand_map)

    arr1 = np.zeros((1080, 1920), dtype=np.uint8)
    for x in range(1920):
        for y in range(1080):
            pose_p = heat_map[y][x]
            face_p = face_map[y][x]
            hand_p = hand_map[y][x]
            ml = [pose_p, face_p, hand_p]
            m = max(ml)
            arr1[y][x] = m

    img = Image.fromarray(arr1)
    img.save(_fp)

    return
