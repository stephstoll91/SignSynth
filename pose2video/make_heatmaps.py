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
        except:
            print("shit")
        for j in range(x[0], x[1] + 1):
            k = inter1(j)
            if not np.isnan(k):
                kk = round(int(k))
                hm[kk][j] = val

        try:
            inter2 = interp1d(y, x, kind='slinear')
        except:
            print("shit")
        for j in range(y[0], y[1] + 1):
            k = inter2(j)
            if not np.isnan(k):
                kk = round(int(k))
                hm[j][kk] = val

        x = [xx[i + 1], xx[i]]
        y = [yy[i + 1], yy[i]]

        try:
            inter1 = interp1d(x, y, kind='slinear')
        except:
            print("shit")
        for j in range(x[0], x[1] + 1):
            k = inter1(j)
            if not np.isnan(k):
                kk = round(int(k))
                hm[kk][j] = val

        try:
            inter2 = interp1d(y, x, kind='slinear')
        except:
            print("shit")
        for j in range(y[0], y[1] + 1):
            k = inter2(j)
            if not np.isnan(k):
                kk = round(int(k))
                hm[j][kk] = val

    return hm
def draw_face(xf, yf, val, fff):
    heat_map = np.zeros((720, 1280), dtype=np.uint8)

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

    img = Image.fromarray(heat_map)
    fp = os.path.splitext(fff)[0] + ".png"
    save_path = os.path.join(png_face_path, fp)
    img.save(save_path)

    return val


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


def get_pickle_paths():
    poses = []
    glosses = []
    poses0 = []
    poses1 = []
    poses2 = []
    poses3 = []
    poses4 = []
    glosses0 = []
    glosses1 = []
    glosses2 = []
    glosses3 = []
    glosses4 = []
    for (dirpath, dirnames, filenames) in os.walk('/vol/vssp/smile/Steph/pycharm_projects/pix2pixHD-master/gen_data/u400/'):
        for d in dirnames:
            files = os.listdir(dirpath + "/" + d)
            for f in files:
                if "_0_pose_data_gen" in f:
                    poses0.append(dirpath + d + "/" + f)
                if "_0_gloss_data_gen" in f:
                    glosses0.append(dirpath + d + "/" + f)
                if "_1_pose_data_gen" in f:
                    poses1.append(dirpath + d + "/" + f)
                if "_1_gloss_data_gen" in f:
                    glosses1.append(dirpath + d + "/" + f)
                if "_2_pose_data_gen" in f:
                    poses2.append(dirpath + d + "/" + f)
                if "_2_gloss_data_gen" in f:
                    glosses2.append(dirpath + d + "/" + f)
                if "_3_pose_data_gen" in f:
                    poses3.append(dirpath + d + "/" + f)
                if "_3_gloss_data_gen" in f:
                    glosses3.append(dirpath + d + "/" + f)
                if "_4_pose_data_gen" in f:
                    poses4.append(dirpath + d + "/" + f)
                if "_4_gloss_data_gen" in f:
                    glosses4.append(dirpath + d + "/" + f)
    poses.append(poses0)
    poses.append(poses1)
    poses.append(poses2)
    poses.append(poses3)
    poses.append(poses4)

    glosses.append(glosses0)
    glosses.append(glosses1)
    glosses.append(glosses2)
    glosses.append(glosses3)
    glosses.append(glosses4)

    return poses, glosses


def load_from_pickle(poses, glosses, ref_x_p, ref_y_p , l):

    x_p = []
    y_p = []
    heads = []
    necks = []
    shoulder_right = []
    elbow_right = []
    wrist_right = []
    shoulder_left = []
    elbow_left = []
    wrist_left = []

    ref_x = np.asarray(ref_x_p)
    ref_y = np.asarray(ref_y_p)

    for pose_p, gloss_p in zip(poses, glosses):
        pose_p_split = pose_p.split("/")
        with open(pose_p, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()

        g = pickle.load(open(gloss_p, "rb"))
        for pp in p:
            pp[:, :2] = np.cumsum(pp[:, :2], axis=0)
            ppp = np.hstack((smoothListGaussian(pp[:, 0]).reshape(-1, 1), smoothListGaussian(pp[:, 1]).reshape(-1, 1)))
            if pose_p_split[-2] == "head":
                heads.append(ppp)
            elif pose_p_split[-2] == "neck":
                necks.append(ppp)
            elif pose_p_split[-2] == "left_shoulder":
                shoulder_left.append(ppp)
            elif pose_p_split[-2] == "left_elbow":
                elbow_left.append(ppp)
            elif pose_p_split[-2] == "left_wrist":
                wrist_left.append(ppp)
            elif pose_p_split[-2] == "right_shoulder":
                shoulder_right.append(ppp)
            elif pose_p_split[-2] == "right_elbow":
                elbow_right.append(ppp)
            elif pose_p_split[-2] == "right_wrist":
                wrist_right.append(ppp)

    for i in range(len(heads)):
        h = heads[i]
        n = necks[i]
        sr = shoulder_right[i]
        er = elbow_right[i]
        wr = wrist_right[i]
        sl = shoulder_left[i]
        el = elbow_left[i]
        wl = wrist_left[i]

        x_p = np.hstack((h[:, 0].reshape(-1,1), n[:, 0].reshape(-1,1), sr[:,0].reshape(-1,1), er[:,0].reshape(-1,1), wr[:,0].reshape(-1,1), sl[:,0].reshape(-1,1), el[:,0].reshape(-1,1), wl[:,0].reshape(-1,1)))
        y_p = np.hstack((h[:, 1].reshape(-1, 1), n[:, 1].reshape(-1, 1), sr[:, 1].reshape(-1, 1),
                         er[:, 1].reshape(-1, 1), wr[:, 1].reshape(-1, 1), sl[:, 1].reshape(-1, 1),
                         el[:, 1].reshape(-1, 1), wl[:, 1].reshape(-1, 1)))
        [u, v] = x_p.shape
        prev_heat = np.zeros((720, 1280), dtype=np.uint8)
        for j in range(u):
            heat_map = np.zeros((720, 1280), dtype=np.uint8)
            weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            xx = x_p[j, :] * weights + ref_x[0:8]
            yy = y_p[j, :] * weights + ref_y[0:8]
            [xn, yn] = align_to_ref(ref_x, ref_y, xx, yy)
            # plt.figure(figsize=(20, 20))
            # plt.scatter(xx, yy)
            # plt.gca().invert_yaxis()
            # path = "/vol/vssp/smile/Steph/pycharm_projects/pix2pixHD-master/gen_data/renormed/" + g[i] + str(l) + "/"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # save_path = path + str(j) + "_test.png"
            # plt.savefig(save_path)
            # plt.clf();
            # plt.cla()
            for x, y in zip(xn, yn):
               if x > 0.0 and y > 0.0:
                    x = int(round(x))
                    y = int(round(y))

                    heat_map[y-10:y+10, x-10:x+10] = 255
            heat_map = heat_map #+ prev_heat
            img = Image.fromarray(heat_map)
            path = "/vol/vssp/smile/Steph/pycharm_projects/pix2pixHD-master/gen_data/u400/" + g[i] + str(l) + "/"
            if not os.path.exists(path):
                os.makedirs(path)
            save_path = path + str(j) + "_test.png"
            img.save(save_path)
            prev_heat += heat_map

    return x_p, y_p

def load_data_from_json(myfile):
    d = myfile.read().replace('\n', '')
    data = json.loads(d)
    if data['people']:
        pose_2d = data['people'][0]['pose_keypoints_2d']
        face_2d = data['people'][0]['face_keypoints_2d']
        hand_left_2d = data['people'][0]['hand_left_keypoints_2d']
        hand_right_2d = data['people'][0]['hand_right_keypoints_2d']

        # pose
        _x_p = pose_2d[::3]
        _y_p = pose_2d[1::3]
        # face
        _x_f = face_2d[::3]
        _y_f = face_2d[1::3]
        # hand left
        _x_hl = hand_left_2d[::3]
        _y_hl = hand_left_2d[1::3]
        # hand right
        _x_hr = hand_right_2d[::3]
        _y_hr = hand_right_2d[1::3]

        # pose keypoints heatmap
        _x_p = [_x_p[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18)]  # x keypoints for upper body
        _y_p = [_y_p[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18)]  # y keypoints for upper body

    return _x_p, _y_p, _x_f, _y_f, _x_hl, _y_hl, _x_hr, _y_hr

def align_to_ref(_ref_x_p, _ref_y_p,_x_p, _y_p):
    ref_neck_x = ref_x_p[1]
    ref_neck_y = ref_y_p[1]
    ref_sr_x = ref_x_p[2]
    ref_sr_y = ref_y_p[2]
    ref_sl_x = ref_x_p[5]
    ref_sl_y = ref_y_p[5]

    ref_ds_x = abs(ref_sl_x - ref_sr_x)
    ref_ds_y = abs(ref_sl_y - ref_sr_y)

    neck_x = _x_p[1]
    neck_y = _y_p[1]
    _sr_x = _x_p[2]
    _sr_y = _y_p[2]
    _sl_x = _x_p[5]
    _sl_y = _y_p[5]

    _ds_x = abs(_sl_x - _sr_x)
    _ds_y = abs(_sl_y - _sr_y)

    diff_neck_x = ref_neck_x - neck_x
    diff_neck_y = ref_neck_y - neck_y

    x_trans = [x+diff_neck_x for x in _x_p]
    y_trans = [y+diff_neck_y for y in _y_p]

    f1 = ref_ds_x /_ds_x
    f2 = ref_ds_y / _ds_y

    xn = []
    yn = []
    for x in x_trans:
        xn.append(ref_neck_x + (x - ref_neck_x) * (f1))
    for y in y_trans:
        yn.append(ref_neck_y + (y - ref_neck_y) * (f1))

    return xn, yn





json_path = "/vol/vssp/smile/Steph/pycharm_projects/openpose_phoenix/test/01April_2010_Thursday_heute-6704/"
png_path = "/vol/vssp/smile/Steph/pycharm_projects/openpose_phoenix/labels/"
png_face_path = "/vol/vssp/smile/Steph/pycharm_projects/openpose_phoenix/label_face/"
png_hands_path = "/vol/vssp/smile/Steph/pycharm_projects/openpose_phoenix/label_hands/"
ref_path = "/vol/vssp/smile/Steph/pycharm_projects/openpose_phoenix/reference_skel_dynavis.json"
if not os.path.exists(png_path):
    os.makedirs(png_path)
if not os.path.exists(png_face_path):
    os.makedirs(png_face_path)
if not os.path.exists(png_hands_path):
    os.makedirs(png_hands_path)

with open(ref_path) as reffile:
    ref_x_p, ref_y_p, ref_x_f, ref_y_f, ref_x_hl, ref_y_hl, ref_x_hr, ref_y_hr = load_data_from_json(reffile)

[gen_poses, gen_glosses] = get_pickle_paths()

k = 0
for poses, glosses in zip(gen_poses, gen_glosses):
     [x_p, y_p] = load_from_pickle(poses, glosses, ref_x_p, ref_y_p, k)
     k += 1


#for (root, dirs, files) in os.walk(json_path):
    #prev_heat = np.zeros((720, 1280), dtype=np.uint8)
    #for f in files:
        #if "1_Overview" in f:
        #ff = os.path.join(root, f)

        #with open(ff) as myfile:

            #x_p, y_p, x_f, y_f, x_hl, y_hl, x_hr, y_hr = load_data_from_json(myfile)

            #x_p, y_p = align_to_ref(ref_x_p, ref_y_p, x_p, y_p)
            #heat_map = np.zeros((720, 1280), dtype=np.uint8)

            #val = 255
            #for x, y in zip(x_p, y_p):
                #if x > 0.0 and y > 0.0:
                    #x = int(round(x))
                    #y = int(round(y))

                    #heat_map[y-10:y+10, x-10:x+10] = val
                #val += 1
            #heat_map = heat_map #+ prev_heat
            #img = Image.fromarray(heat_map)
            #fp = os.path.splitext(f)[0] + ".png"
            #save_path = os.path.join(png_path, fp)
            #img.save(save_path)
            #prev_heat += heat_map
            # # face heatmap
            # val = 16
            # val = draw_face(x_f, y_f, val, f)
            #
            # #val = 255
            #
            # heat_map = np.zeros((720, 1280), dtype=np.uint8)
            # heat_map = draw_hand(x_hl, y_hl, val, heat_map)
            # val += 1
            # heat_map = draw_hand(x_hr, y_hr, val, heat_map)
            #
            # img = Image.fromarray(heat_map)
            # fp = os.path.splitext(f)[0] + ".png"
            # save_path = os.path.join(png_hands_path, fp)
            # img.save(save_path)




